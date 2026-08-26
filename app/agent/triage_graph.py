"""使用 LangGraph 编排景区服务诉求的演示处理流程。"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.result_parser import (
    build_knowledge_miss_result,
    build_provider_error_result,
    parse_service_case_result,
)
from app.llm.provider import (
    ChatMessage,
    LLMProvider,
    ModelProviderError,
    StructuredGenerationRequest,
)
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import (
    KnowledgeReference,
    ServiceCaseResult,
    ServiceRequestInput,
)


class TriageGraphState(TypedDict, total=False):
    """LangGraph 节点之间传递的内部状态，不直接作为 API 输出。"""

    service_request: ServiceRequestInput
    retrieved_references: list[KnowledgeReference]
    retrieval_error: str | None
    raw_model_output: str
    provider_error: str | None
    result: ServiceCaseResult


class TriageAgent:
    """景区服务文本 Agent 的可调用入口，固定返回 Pydantic 结果。"""

    def __init__(
        self,
        knowledge_store: ChromaKnowledgeStore,
        model_provider: LLMProvider,
    ) -> None:
        self._graph = build_triage_graph(
            knowledge_store=knowledge_store,
            model_provider=model_provider,
        )

    def run(self, service_request: ServiceRequestInput) -> ServiceCaseResult:
        """执行完整工作流；图本身异常时也返回保守的人工复核结果。"""

        try:
            state = self._graph.invoke({"service_request": service_request})
        except Exception as error:
            return build_knowledge_miss_result(
                service_request=service_request,
                retrieval_error=f"agent_graph_failed: {error}",
            )

        result = state.get("result")
        if isinstance(result, ServiceCaseResult):
            return result

        return build_knowledge_miss_result(
            service_request=service_request,
            retrieval_error="agent_graph_finished_without_a_valid_result",
        )


def build_triage_graph(
    *,
    knowledge_store: ChromaKnowledgeStore,
    model_provider: LLMProvider,
):
    """构建检索、模型生成和安全解析三个节点组成的 LangGraph。"""

    def retrieve_knowledge(state: TriageGraphState) -> dict[str, object]:
        """先检索本地演示资料；检索异常也要让后续流程安全结束。"""

        service_request = state["service_request"]
        try:
            references = knowledge_store.search(service_request.text, limit=3)
            return {"retrieved_references": references, "retrieval_error": None}
        except Exception as error:
            return {
                "retrieved_references": [],
                "retrieval_error": f"knowledge_store_error: {error}"[:500],
            }

    def generate_model_output(state: TriageGraphState) -> dict[str, object]:
        """仅在知识命中后调用模型，并始终要求结构化 JSON 输出。"""

        service_request = state["service_request"]
        references = state["retrieved_references"]
        try:
            response = model_provider.generate_json(
                _build_model_request(service_request, references)
            )
            return {"raw_model_output": response.content, "provider_error": None}
        except ModelProviderError as error:
            return {"provider_error": str(error)[:500]}
        except Exception as error:
            # 即使提供方实现出现意外异常，也不允许工作流直接返回未校验内容。
            return {"provider_error": f"unexpected_provider_error: {error}"[:500]}

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
        return {"result": result}

    def build_knowledge_review(state: TriageGraphState) -> dict[str, ServiceCaseResult]:
        """知识未命中时不调用模型，直接生成必须人工复核的安全结果。"""

        return {
            "result": build_knowledge_miss_result(
                service_request=state["service_request"],
                retrieval_error=state.get("retrieval_error"),
            )
        }

    graph = StateGraph(TriageGraphState)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("generate_model_output", generate_model_output)
    graph.add_node("parse_model_output", parse_model_output)
    graph.add_node("build_knowledge_review", build_knowledge_review)

    graph.add_edge(START, "retrieve_knowledge")
    graph.add_conditional_edges(
        "retrieve_knowledge",
        _route_after_retrieval,
        {
            "generate_model_output": "generate_model_output",
            "build_knowledge_review": "build_knowledge_review",
        },
    )
    graph.add_edge("generate_model_output", "parse_model_output")
    graph.add_edge("parse_model_output", END)
    graph.add_edge("build_knowledge_review", END)
    return graph.compile()


def _route_after_retrieval(state: TriageGraphState) -> str:
    """知识库异常或未命中时，跳过模型调用以控制成本并保证人工复核。"""

    if state.get("retrieval_error") or not state.get("retrieved_references"):
        return "build_knowledge_review"
    return "generate_model_output"


def _build_model_request(
    service_request: ServiceRequestInput,
    references: list[KnowledgeReference],
) -> StructuredGenerationRequest:
    """创建传给任意 OpenAI-compatible 提供方的结构化请求。"""

    source_payload = [reference.model_dump(mode="json") for reference in references]
    user_payload = {
        "service_request": service_request.model_dump(mode="json"),
        "retrieved_references": source_payload,
    }
    system_prompt = (
        "你正在处理仅供项目演示的景区文本诉求。"
        "只能返回与给定 JSON Schema 匹配的 JSON 对象，不要输出 Markdown。"
        "只能引用 retrieved_references 中完全一致的资料；"
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
