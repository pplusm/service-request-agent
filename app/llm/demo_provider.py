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
from app.rules.scenic_service_config import (
    DemonstrationRoute,
    EventMatcher,
    RiskRule,
    ScenicServiceConfiguration,
    load_scenic_service_configuration,
)
from app.schemas.models import (
    KnowledgeReference,
    ReviewReason,
    RiskLevel,
    ServiceRequestInput,
)


# 以下词语只用于项目教学中的确定性文本匹配，不是实际语言模型能力。
_DEMO_LOCATION_PATTERN: Final = re.compile(
    r"(?P<location>(?:东|西|南|北)门(?:附近)?|游客中心(?:附近)?|"
    r"停车场(?:附近)?|入口(?:附近)?|出口(?:附近)?|服务台(?:附近)?|"
    r"观景台(?:附近)?|广场(?:附近)?)"
)
class DemoLLMProvider(LLMProvider):
    """不联网的确定性演示提供方，仅用于本项目的本地展示和测试。"""

    def __init__(
        self,
        *,
        configuration: ScenicServiceConfiguration | None = None,
        simulate_invalid_output: bool = False,
    ) -> None:
        # 该开关专门用于演示模型输出不符合 JSON Schema 时的安全兜底路径。
        self._simulate_invalid_output = simulate_invalid_output

        # 默认从 YAML 加载配置；测试也可以注入临时配置验证不再依赖硬编码规则。
        self._configuration = configuration or load_scenic_service_configuration()

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
                configuration=self._configuration,
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
    configuration: ScenicServiceConfiguration,
) -> str:
    """按可控演示规则生成结果；每份引用都来自本次实际检索。"""

    event_matcher = configuration.find_event_matcher(service_request.text)
    if event_matcher is None or not references:
        # 相近资料不足以支持请求时，不能保留它作为本案依据。
        return _build_unsupported_issue_response(service_request)

    location = _extract_demo_location(service_request.text)
    high_risk_rule = configuration.find_high_risk_rule(service_request.text)
    if high_risk_rule is not None:
        return _build_high_risk_response(
            service_request=service_request,
            reference=references[0],
            location=location,
            event_matcher=event_matcher,
            risk_rule=high_risk_rule,
        )

    route = configuration.find_route(
        event_type=event_matcher.event_type,
        risk_level=RiskLevel.LOW,
    )
    if route is None:
        # 配置没有为当前低风险事件提供建议时，宁可转人工，也不能猜测动作。
        return _build_unsupported_issue_response(service_request)

    return _build_facility_fault_response(
        service_request=service_request,
        reference=references[0],
        location=location,
        event_matcher=event_matcher,
        route=route,
    )


def _extract_demo_location(text: str) -> str | None:
    """从有限的演示地点词中提取位置；未找到时要求人工确认。"""

    match = _DEMO_LOCATION_PATTERN.search(text)
    if match is None:
        return None
    return match.group("location")


def _build_facility_fault_response(
    *,
    service_request: ServiceRequestInput,
    reference: KnowledgeReference,
    location: str | None,
    event_matcher: EventMatcher,
    route: DemonstrationRoute,
) -> str:
    """构造普通设施问题的结果；地点缺失时禁止自动给出建议。"""

    entity_values = {
        "location": location,
        "facility_name": event_matcher.facility_name,
        "visitor_condition": None,
        "estimated_affected_count": None,
        "event_time_description": None,
    }
    missing_fields = [
        field_name
        for field_name in route.required_fields
        if entity_values[field_name] is None
    ]
    source_matches_route = reference.source_id == route.knowledge_source_id
    requires_human_review = bool(missing_fields) or not source_matches_route
    action_plan: list[dict[str, object]] = []
    if not requires_human_review:
        action_plan.append(
            {
                "step": 1,
                "suggested_action": route.suggested_action,
                "knowledge_source_ids": [route.knowledge_source_id],
                "is_demo_action": route.is_demo_action,
            }
        )

    review_reasons: list[str] = []
    if missing_fields:
        review_reasons.append(ReviewReason.MISSING_FIELDS.value)
    if not source_matches_route:
        review_reasons.append(ReviewReason.RULE_CONFLICT.value)

    if missing_fields:
        review_note = "配置要求的字段缺失，需人工补充确认。"
    elif not source_matches_route:
        review_note = "演示建议配置与本次知识来源不一致，已转人工复核。"
    else:
        review_note = ""

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": event_matcher.facility_name,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": event_matcher.event_type.value,
            # 这是固定演示分数，不是系统准确率或真实模型指标。
            "confidence": 0.9,
            "evidence": [service_request.text],
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
            "reasons": review_reasons,
            "review_note": review_note,
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
    event_matcher: EventMatcher,
    risk_rule: RiskRule,
) -> str:
    """高风险提示出现时保留人工复核，且不输出自动处置建议。"""

    missing_fields = [] if location is not None else ["location"]
    review_reasons = [risk_rule.review_reason.value]
    if missing_fields:
        review_reasons.append("missing_fields")

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": event_matcher.facility_name,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": event_matcher.event_type.value,
            "confidence": 0.9,
            "evidence": [service_request.text],
        },
        risk={
            "level": risk_rule.risk_level.value,
            "risk_factors": [f"命中演示风险规则：{risk_rule.rule_id}"],
            "summary": risk_rule.risk_summary,
        },
        knowledge_references=[reference.model_dump(mode="json")],
        action_plan=[],
        review={
            "requires_human_review": True,
            "reasons": review_reasons,
            "review_note": risk_rule.review_note,
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
