"""提供命令行和 API 共用的免费本地演示模型。"""

from __future__ import annotations

import json
import re
from typing import Final

from pydantic import ValidationError

from app.llm.provider import (
    LLMProvider,
    LLMResponse,
    ModelProviderError,
    StructuredGenerationRequest,
)
from app.schemas.models import KnowledgeReference, ServiceRequestInput


# 以下词语只用于项目教学中的确定性文本匹配，不是实际语言模型能力。
_DEMO_LOCATION_PATTERN: Final = re.compile(
    r"(?P<location>(?:东|西|南|北)门(?:附近)?|游客中心(?:附近)?|"
    r"停车场(?:附近)?|入口(?:附近)?|出口(?:附近)?|服务台(?:附近)?|"
    r"观景台(?:附近)?|广场(?:附近)?)"
)
_HIGH_RISK_KEYWORDS: Final = (
    "受伤",
    "昏倒",
    "晕倒",
    "摔倒",
    "流血",
    "火灾",
    "起火",
    "烟雾",
    "危险",
    "急救",
)


class DemoLLMProvider(LLMProvider):
    """不联网的确定性演示提供方，仅用于本项目的本地展示和测试。"""

    def __init__(self, *, simulate_invalid_output: bool = False) -> None:
        # 该开关专门用于演示模型输出不符合 JSON Schema 时的安全兜底路径。
        self._simulate_invalid_output = simulate_invalid_output

        # 保留调用快照，便于测试确认 Agent 向提供方传递了结构化请求。
        self.requests: list[StructuredGenerationRequest] = []

    def generate_json(
        self, request: StructuredGenerationRequest
    ) -> LLMResponse:
        """根据 Agent 提供的结构化上下文返回模拟原始 JSON 文本。"""

        self.requests.append(request.model_copy(deep=True))
        if self._simulate_invalid_output:
            return self._response("{}")

        service_request, references = _read_agent_context(request)
        return self._response(
            _build_demo_response(
                service_request=service_request,
                references=references,
            )
        )

    @staticmethod
    def _response(content: str) -> LLMResponse:
        """补充演示提供方标识，原始内容仍交给 Agent 的解析器校验。"""

        return LLMResponse(
            content=content,
            provider_name="local_demo",
            model_name="deterministic-demo-v1",
        )


def _read_agent_context(
    request: StructuredGenerationRequest,
) -> tuple[ServiceRequestInput, list[KnowledgeReference]]:
    """从统一提供方请求中恢复 Agent 已传入的案件和检索来源。"""

    user_message = next(
        (message for message in reversed(request.messages) if message.role == "user"),
        None,
    )
    if user_message is None:
        raise ModelProviderError("demo provider request has no user message")

    try:
        payload = json.loads(user_message.content)
        if not isinstance(payload, dict):
            raise ValueError("demo provider payload must be an object")

        raw_references = payload.get("retrieved_references")
        if not isinstance(raw_references, list):
            raise ValueError("demo provider references must be a list")

        service_request = ServiceRequestInput.model_validate(
            payload.get("service_request")
        )
        references = [
            KnowledgeReference.model_validate(reference)
            for reference in raw_references
        ]
        return service_request, references
    except (TypeError, ValueError, ValidationError) as error:
        raise ModelProviderError(
            f"demo_provider_request_invalid: {error}"[:500]
        ) from error


def _build_demo_response(
    *,
    service_request: ServiceRequestInput,
    references: list[KnowledgeReference],
) -> str:
    """按可控演示规则生成结果；每份引用都来自本次实际检索。"""

    issue = _identify_demo_issue(service_request.text)
    if issue is None or not references:
        # 相近资料不足以支持请求时，不能保留它作为本案依据。
        return _build_unsupported_issue_response(service_request)

    facility_name, evidence = issue
    location = _extract_demo_location(service_request.text)
    if _contains_high_risk_signal(service_request.text):
        return _build_high_risk_response(
            service_request=service_request,
            reference=references[0],
            location=location,
            facility_name=facility_name,
            evidence=evidence,
        )

    return _build_facility_fault_response(
        service_request=service_request,
        reference=references[0],
        location=location,
        facility_name=facility_name,
        evidence=evidence,
    )


def _identify_demo_issue(text: str) -> tuple[str, str] | None:
    """识别演示资料列出的设施问题，并保留用户输入作为证据。"""

    normalized_text = text.strip()
    if ("卫生间" in normalized_text or "洗手间" in normalized_text or "厕所" in normalized_text) and (
        "没水" in normalized_text
        or "无水" in normalized_text
        or "停水" in normalized_text
    ):
        return "卫生间", normalized_text

    if "指示牌" in normalized_text and (
        "损坏" in normalized_text
        or "破损" in normalized_text
        or "坏了" in normalized_text
    ):
        return "指示牌", normalized_text

    if ("照明" in normalized_text or "路灯" in normalized_text) and (
        "故障" in normalized_text
        or "损坏" in normalized_text
        or "不亮" in normalized_text
        or "坏了" in normalized_text
    ):
        return "照明设施", normalized_text

    return None


def _extract_demo_location(text: str) -> str | None:
    """从有限的演示地点词中提取位置；未找到时要求人工确认。"""

    match = _DEMO_LOCATION_PATTERN.search(text)
    if match is None:
        return None
    return match.group("location")


def _contains_high_risk_signal(text: str) -> bool:
    """识别演示高风险提示词；该判断不构成真实安全评估。"""

    return any(keyword in text for keyword in _HIGH_RISK_KEYWORDS)


def _build_facility_fault_response(
    *,
    service_request: ServiceRequestInput,
    reference: KnowledgeReference,
    location: str | None,
    facility_name: str,
    evidence: str,
) -> str:
    """构造普通设施问题的结果；地点缺失时禁止自动给出建议。"""

    missing_fields = [] if location is not None else ["location"]
    requires_human_review = bool(missing_fields)
    action_plan: list[dict[str, object]] = []
    if not requires_human_review:
        action_plan.append(
            {
                "step": 1,
                "suggested_action": "创建演示性设施维护跟进建议。",
                "knowledge_source_ids": [reference.source_id],
                "is_demo_action": True,
            }
        )

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": facility_name,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": "facility_fault",
            # 这是固定演示分数，不是系统准确率或真实模型指标。
            "confidence": 0.9,
            "evidence": [evidence],
        },
        risk={
            "level": "low",
            "risk_factors": [],
            "summary": "仅用于项目演示的低风险设施故障判断。",
        },
        knowledge_references=[reference.model_dump(mode="json")],
        action_plan=action_plan,
        review={
            "requires_human_review": requires_human_review,
            "reasons": ["missing_fields"] if requires_human_review else [],
            "review_note": "地点信息缺失，需人工补充确认。"
            if requires_human_review
            else "",
        },
        diagnostics={
            "knowledge_hit": True,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    )


def _build_high_risk_response(
    *,
    service_request: ServiceRequestInput,
    reference: KnowledgeReference,
    location: str | None,
    facility_name: str,
    evidence: str,
) -> str:
    """高风险提示出现时保留人工复核，且不输出自动处置建议。"""

    missing_fields = [] if location is not None else ["location"]
    review_reasons = ["high_risk"]
    if missing_fields:
        review_reasons.append("missing_fields")

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": facility_name,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": "facility_fault",
            "confidence": 0.9,
            "evidence": [evidence],
        },
        risk={
            "level": "high",
            "risk_factors": ["文本含有可能的安全或健康风险描述。"],
            "summary": "演示性高风险提示，必须由人工确认。",
        },
        knowledge_references=[reference.model_dump(mode="json")],
        action_plan=[],
        review={
            "requires_human_review": True,
            "reasons": review_reasons,
            "review_note": "检测到可能的高风险描述，已转人工复核。",
        },
        diagnostics={
            "knowledge_hit": True,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    )


def _build_unsupported_issue_response(
    service_request: ServiceRequestInput,
) -> str:
    """资料不足时返回可解析的保守人工复核结果。"""

    location = _extract_demo_location(service_request.text)
    missing_fields = ["facility_name"]
    if location is None:
        missing_fields.insert(0, "location")

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": None,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": "other_unknown",
            "confidence": 0.0,
            "evidence": [],
        },
        risk={
            "level": "unassessed",
            "risk_factors": [],
            "summary": "当前无法安全评估风险，需人工复核。",
        },
        # 检索到的相近资料未被用作依据，因此结果明确标记为知识未命中。
        knowledge_references=[],
        action_plan=[],
        review={
            "requires_human_review": True,
            "reasons": [
                "unassessed_risk",
                "missing_fields",
                "knowledge_not_found",
                "low_confidence",
            ],
            "review_note": "演示资料未明确支持该诉求，已转人工复核。",
        },
        diagnostics={
            "knowledge_hit": False,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    )


def _serialize_result_payload(
    *,
    service_request: ServiceRequestInput,
    entities: dict[str, object],
    classification: dict[str, object],
    risk: dict[str, object],
    knowledge_references: list[dict[str, object]],
    action_plan: list[dict[str, object]],
    review: dict[str, object],
    diagnostics: dict[str, object],
) -> str:
    """统一序列化模拟原始输出，随后仍由 Agent 的 Pydantic 解析器校验。"""

    payload = {
        "request_id": service_request.request_id,
        "scenario": service_request.scenario.value,
        "entities": entities,
        "classification": classification,
        "risk": risk,
        "knowledge_references": knowledge_references,
        "action_plan": action_plan,
        "review": review,
        "diagnostics": diagnostics,
    }
    return json.dumps(payload, ensure_ascii=False)
