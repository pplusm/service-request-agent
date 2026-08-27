"""使用 LangGraph 编排景区服务诉求的演示处理流程。"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agent.result_parser import (
    attach_vision_context,
    build_knowledge_miss_result,
    build_provider_error_result,
    build_vision_error_result,
    parse_service_case_result,
    redact_image_payload_from_diagnostics,
)
from app.agent.multimodal_fusion import build_multimodal_fusion
from app.llm.provider import (
    ChatMessage,
    LLMProvider,
    ModelProviderError,
    StructuredGenerationRequest,
)
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.rules.risk_enforcement import enforce_configured_high_risk_rule
from app.rules.scenic_service_config import (
    ScenicServiceConfiguration,
    load_scenic_service_configuration,
)
from app.schemas.models import (
    KnowledgeReference,
    MultimodalFusion,
    ReviewReason,
    ServiceCaseResult,
    ServiceRequestInput,
    VisionObservation,
)
from app.vision.factory import build_vision_provider_from_environment
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProvider,
    VisionProviderError,
)


class TriageGraphState(TypedDict, total=False):
    """LangGraph 节点之间传递的内部状态，不直接作为 API 输出。"""

    service_request: ServiceRequestInput
    retrieved_references: list[KnowledgeReference]
    retrieval_error: str | None
    raw_model_output: str
    provider_error: str | None
    vision_observation: VisionObservation | None
    vision_error: str | None
    raw_vision_output: str | None
    vision_call_success: bool | None
    vision_output_parse_success: bool | None
    vision_provider_name: str | None
    vision_model_name: str | None
    multimodal_fusion: MultimodalFusion | None
    result: ServiceCaseResult


class TriageAgent:
    """景区服务文本 Agent 的可调用入口，固定返回 Pydantic 结果。"""

    def __init__(
        self,
        knowledge_store: ChromaKnowledgeStore,
        model_provider: LLMProvider,
        vision_provider: VisionProvider | None = None,
        configuration: ScenicServiceConfiguration | None = None,
    ) -> None:
        # 风险配置由 Agent 持有，确保未来替换真实模型后仍保留规则兜底。
        self._configuration = configuration or load_scenic_service_configuration()
        # 默认使用不联网的视觉演示模型；后续可注入任意 VisionProvider 实现。
        self._vision_provider = (
            vision_provider or build_vision_provider_from_environment()
        )
        self._graph = build_triage_graph(
            knowledge_store=knowledge_store,
            model_provider=model_provider,
            vision_provider=self._vision_provider,
            configuration=self._configuration,
        )

    def run(self, service_request: ServiceRequestInput) -> ServiceCaseResult:
        """执行完整工作流；图本身异常时也返回保守的人工复核结果。"""

        try:
            state = self._graph.invoke({"service_request": service_request})
            result = state.get("result")
        except Exception as error:
            return self._build_graph_fallback_result(
                service_request=service_request,
                failure_detail=f"agent_graph_failed: {error}",
            )

        if isinstance(result, ServiceCaseResult):
            return result

        return self._build_graph_fallback_result(
            service_request=service_request,
            failure_detail="agent_graph_finished_without_a_valid_result",
        )

    def _build_graph_fallback_result(
        self,
        *,
        service_request: ServiceRequestInput,
        failure_detail: str,
    ) -> ServiceCaseResult:
        """图级失败时保留图片摘要，并明确要求人工检查视觉流程。"""

        result = build_knowledge_miss_result(
            service_request=service_request,
            retrieval_error=failure_detail,
        )
        if service_request.image is not None:
            # 图在任何节点中断时无法确认视觉状态，因此不伪造调用结果，只要求复核。
            result = attach_vision_context(
                result=result,
                service_request=service_request,
                observation=None,
                vision_call_success=None,
                vision_output_parse_success=None,
                raw_vision_output=None,
                vision_provider_name=None,
                vision_model_name=None,
                failure_reason=ReviewReason.VISION_ERROR,
            )

        return enforce_configured_high_risk_rule(
            result=result,
            service_request=service_request,
            configuration=self._configuration,
        )


def build_triage_graph(
    *,
    knowledge_store: ChromaKnowledgeStore,
    model_provider: LLMProvider,
    vision_provider: VisionProvider,
    configuration: ScenicServiceConfiguration,
):
    """构建视觉抽取、检索、模型生成和安全解析节点组成的 LangGraph。"""

    def vision_extract(state: TriageGraphState) -> dict[str, object]:
        """有图片时调用视觉提供方，并把输出限制为结构化观察。"""

        service_request = state["service_request"]
        if service_request.image is None:
            # 没有图片时完全跳过视觉模型，保持文本 MVP 的调用成本和行为。
            return {
                "vision_observation": None,
                "vision_error": None,
                "raw_vision_output": None,
                "vision_call_success": None,
                "vision_output_parse_success": None,
                "vision_provider_name": None,
                "vision_model_name": None,
                "multimodal_fusion": None,
            }

        try:
            response = vision_provider.generate_json(
                _build_vision_request(service_request)
            )
        except VisionProviderError as error:
            return {
                "vision_observation": None,
                "vision_error": redact_image_payload_from_diagnostics(
                    str(error),
                    service_request,
                    max_length=500,
                ),
                "raw_vision_output": None,
                "vision_call_success": False,
                "vision_output_parse_success": None,
                "vision_provider_name": None,
                "vision_model_name": None,
                "multimodal_fusion": None,
            }
        except Exception as error:
            # 不把未知异常直接暴露给调用方，统一转成视觉人工复核。
            return {
                "vision_observation": None,
                "vision_error": redact_image_payload_from_diagnostics(
                    f"unexpected_vision_provider_error: {error}",
                    service_request,
                    max_length=500,
                ),
                "raw_vision_output": None,
                "vision_call_success": False,
                "vision_output_parse_success": None,
                "vision_provider_name": None,
                "vision_model_name": None,
                "multimodal_fusion": None,
            }

        try:
            observation = VisionObservation.model_validate_json(response.content)
        except (ValidationError, ValueError) as error:
            return {
                "vision_observation": None,
                "vision_error": redact_image_payload_from_diagnostics(
                    f"vision_output_validation_failed: {error}",
                    service_request,
                    max_length=500,
                ),
                "raw_vision_output": redact_image_payload_from_diagnostics(
                    response.content,
                    service_request,
                    max_length=5_000,
                ),
                "vision_call_success": True,
                "vision_output_parse_success": False,
                "vision_provider_name": response.provider_name,
                "vision_model_name": response.model_name,
                "multimodal_fusion": None,
            }

        return {
            "vision_observation": observation,
            "vision_error": None,
            "raw_vision_output": None,
            "vision_call_success": True,
            "vision_output_parse_success": True,
            "vision_provider_name": response.provider_name,
            "vision_model_name": response.model_name,
            # 融合只接收已通过 Pydantic 校验的视觉观察，不直接读取图片 Base64。
            "multimodal_fusion": build_multimodal_fusion(
                request_text=service_request.text,
                observation=observation,
            ),
        }

    def retrieve_knowledge(state: TriageGraphState) -> dict[str, object]:
        """先检索本地演示资料；检索异常也要让后续流程安全结束。"""

        service_request = state["service_request"]
        try:
            references = knowledge_store.search(service_request.text, limit=3)
            return {"retrieved_references": references, "retrieval_error": None}
        except Exception as error:
            return {
                "retrieved_references": [],
                "retrieval_error": redact_image_payload_from_diagnostics(
                    f"knowledge_store_error: {error}",
                    service_request,
                    max_length=500,
                ),
            }

    def generate_model_output(state: TriageGraphState) -> dict[str, object]:
        """仅在知识命中后调用模型，并始终要求结构化 JSON 输出。"""

        service_request = state["service_request"]
        references = state["retrieved_references"]
        try:
            response = model_provider.generate_json(
                _build_model_request(
                    service_request,
                    references,
                    state.get("vision_observation"),
                )
            )
            return {"raw_model_output": response.content, "provider_error": None}
        except ModelProviderError as error:
            return {
                "provider_error": redact_image_payload_from_diagnostics(
                    str(error),
                    service_request,
                    max_length=500,
                )
            }
        except Exception as error:
            # 即使提供方实现出现意外异常，也不允许工作流直接返回未校验内容。
            return {
                "provider_error": redact_image_payload_from_diagnostics(
                    f"unexpected_provider_error: {error}",
                    service_request,
                    max_length=500,
                )
            }

    def parse_model_output(state: TriageGraphState) -> dict[str, ServiceCaseResult]:
        """将模型文本交给 Pydantic 校验器，失败时由兜底函数转人工复核。"""

        service_request = state["service_request"]
        references = state["retrieved_references"]
        provider_error = state.get("provider_error")
        if provider_error:
            result = build_provider_error_result(
                service_request=service_request,
                retrieved_references=references,
                provider_error=provider_error,
            )
        else:
            result = parse_service_case_result(
                service_request=service_request,
                raw_model_output=state.get("raw_model_output", ""),
                retrieved_references=references,
            )
        result = attach_vision_context(
            result=result,
            service_request=service_request,
            observation=state.get("vision_observation"),
            vision_call_success=state.get("vision_call_success"),
            vision_output_parse_success=state.get("vision_output_parse_success"),
            raw_vision_output=state.get("raw_vision_output"),
            vision_provider_name=state.get("vision_provider_name"),
            vision_model_name=state.get("vision_model_name"),
            fusion=state.get("multimodal_fusion"),
        )
        return {
            "result": enforce_configured_high_risk_rule(
                result=result,
                service_request=service_request,
                configuration=configuration,
            )
        }

    def build_knowledge_review(state: TriageGraphState) -> dict[str, ServiceCaseResult]:
        """知识未命中时不调用模型，直接生成必须人工复核的安全结果。"""

        service_request = state["service_request"]
        result = build_knowledge_miss_result(
            service_request=service_request,
            retrieval_error=state.get("retrieval_error"),
        )
        result = attach_vision_context(
            result=result,
            service_request=service_request,
            observation=state.get("vision_observation"),
            vision_call_success=state.get("vision_call_success"),
            vision_output_parse_success=state.get("vision_output_parse_success"),
            raw_vision_output=state.get("raw_vision_output"),
            vision_provider_name=state.get("vision_provider_name"),
            vision_model_name=state.get("vision_model_name"),
            fusion=state.get("multimodal_fusion"),
        )
        return {
            "result": enforce_configured_high_risk_rule(
                result=result,
                service_request=service_request,
                configuration=configuration,
            )
        }

    def build_vision_review(state: TriageGraphState) -> dict[str, ServiceCaseResult]:
        """视觉调用或解析失败时，保留图片摘要并直接转人工复核。"""

        service_request = state["service_request"]
        parse_success = state.get("vision_output_parse_success")
        result = build_vision_error_result(
            service_request=service_request,
            retrieved_references=state.get("retrieved_references", []),
            vision_output_parse_success=parse_success,
            raw_vision_output=state.get("raw_vision_output"),
            vision_error=state.get("vision_error", "unknown vision error"),
            vision_call_success=state.get("vision_call_success") is True,
            vision_provider_name=state.get("vision_provider_name"),
            vision_model_name=state.get("vision_model_name"),
        )
        return {
            "result": enforce_configured_high_risk_rule(
                result=result,
                service_request=service_request,
                configuration=configuration,
            )
        }

    graph = StateGraph(TriageGraphState)
    graph.add_node("vision_extract", vision_extract)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("generate_model_output", generate_model_output)
    graph.add_node("parse_model_output", parse_model_output)
    graph.add_node("build_knowledge_review", build_knowledge_review)
    graph.add_node("build_vision_review", build_vision_review)

    graph.add_edge(START, "vision_extract")
    graph.add_edge("vision_extract", "retrieve_knowledge")
    graph.add_conditional_edges(
        "retrieve_knowledge",
        _route_after_retrieval,
        {
            "generate_model_output": "generate_model_output",
            "build_knowledge_review": "build_knowledge_review",
            "build_vision_review": "build_vision_review",
        },
    )
    graph.add_edge("generate_model_output", "parse_model_output")
    graph.add_edge("parse_model_output", END)
    graph.add_edge("build_knowledge_review", END)
    graph.add_edge("build_vision_review", END)
    return graph.compile()


def _route_after_retrieval(state: TriageGraphState) -> str:
    """视觉失败优先转人工，其次处理知识库异常或未命中。"""

    if state.get("vision_error"):
        return "build_vision_review"

    if state.get("retrieval_error") or not state.get("retrieved_references"):
        return "build_knowledge_review"
    return "generate_model_output"


def _build_model_request(
    service_request: ServiceRequestInput,
    references: list[KnowledgeReference],
    vision_observation: VisionObservation | None = None,
) -> StructuredGenerationRequest:
    """创建传给任意 OpenAI-compatible 提供方的结构化请求。"""

    source_payload = [reference.model_dump(mode="json") for reference in references]
    # 绝不把图片 base64 复制给文本模型或写入日志，只传安全元数据。
    service_payload = service_request.model_dump(mode="json", exclude={"image"})
    if service_request.image is not None:
        service_payload["image"] = service_request.image.metadata().model_dump(
            mode="json"
        )
    user_payload = {
        "service_request": service_payload,
        "retrieved_references": source_payload,
        "vision_observation": (
            vision_observation.model_dump(mode="json")
            if vision_observation is not None
            else None
        ),
    }
    system_prompt = (
        "你正在处理仅供项目演示的景区文本诉求。"
        "只能返回与给定 JSON Schema 匹配的 JSON 对象，不要输出 Markdown。"
        "只能引用 retrieved_references 中完全一致的资料；"
        "视觉观察只是辅助证据，不能覆盖文本；图文冲突时必须设置人工复核；"
        "不得编造真实服务规则、真实处理角色、服务时限或准确率。"
        "高风险、信息缺失或无法判断的情况必须设置 requires_human_review=true。"
        "成功返回模型结果时，diagnostics.model_call_success 和 "
        "diagnostics.model_output_parse_success 都必须为 true。"
    )
    return StructuredGenerationRequest(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(
                role="user",
                content=json.dumps(user_payload, ensure_ascii=False),
            ),
        ],
        schema_name="service_case_result",
        json_schema=ServiceCaseResult.model_json_schema(),
        temperature=0.0,
    )


def _build_vision_request(
    service_request: ServiceRequestInput,
) -> VisionExtractionRequest:
    """构造受控视觉抽取请求；调用方只会在图片存在时使用它。"""

    if service_request.image is None:
        raise ValueError("vision request requires an image")
    return VisionExtractionRequest(
        image=service_request.image,
        prompt=(
            "请只返回符合 JSON Schema 的结构化图片观察。"
            "仅记录可见对象、可见文字、地点或设施提示、风险信号和不确定性；"
            "看不清或无法确认的内容必须留在 uncertainty_notes，不能猜测。"
        ),
        schema_name="vision_observation",
        json_schema=VisionObservation.model_json_schema(),
    )
