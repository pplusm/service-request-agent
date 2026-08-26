"""验证模型输出解析和人工复核兜底逻辑。"""

import json

from app.agent.result_parser import build_knowledge_miss_result, parse_service_case_result
from app.llm.mock_provider import MockLLMProvider
from app.llm.provider import ChatMessage, StructuredGenerationRequest
from app.schemas.models import (
    KnowledgeReference,
    ReviewReason,
    RiskLevel,
    ServiceRequestInput,
)


def build_service_request() -> ServiceRequestInput:
    """构造一条固定的演示诉求。"""

    return ServiceRequestInput(
        request_id="demo_001",
        text="东门附近卫生间没水。",
    )


def build_retrieved_reference() -> KnowledgeReference:
    """构造一份模拟 RAG 已实际检索到的演示资料。"""

    return KnowledgeReference(
        source_id="demo_facility_001",
        source_title="Demo facility guidance",
        source_path="data/scenic_service/knowledge/demo_facility.md",
        excerpt="演示资料：卫生间没水属于设施故障示例。",
        relevance_score=0.9,
        is_demo_source=True,
    )


def build_valid_result_payload() -> dict[str, object]:
    """构造符合 ServiceCaseResult 要求的演示模型 JSON。"""

    reference = build_retrieved_reference()
    return {
        "request_id": "demo_001",
        "scenario": "scenic_service",
        "entities": {
            "location": "东门附近",
            "facility_name": "卫生间",
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": [],
        },
        "classification": {
            "event_type": "facility_fault",
            "confidence": 0.9,
            "evidence": ["卫生间没水"],
        },
        "risk": {
            "level": "low",
            "risk_factors": [],
            "summary": "演示性低风险设施故障。",
        },
        "knowledge_references": [reference.model_dump(mode="json")],
        "action_plan": [
            {
                "step": 1,
                "suggested_action": "创建演示性设施维护跟进建议。",
                "knowledge_source_ids": ["demo_facility_001"],
                "is_demo_action": True,
            }
        ],
        "review": {
            "requires_human_review": False,
            "reasons": [],
            "review_note": "",
        },
        "diagnostics": {
            "knowledge_hit": True,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    }


def generate_with_mock(response_content: str) -> str:
    """通过本地模拟模型返回指定文本，证明测试不依赖真实 API。"""

    provider = MockLLMProvider(response_content=response_content)
    response = provider.generate_json(
        StructuredGenerationRequest(
            messages=[ChatMessage(role="user", content="东门附近卫生间没水")],
            schema_name="service_case_result",
            json_schema={"type": "object"},
        )
    )
    return response.content


def test_parser_returns_valid_model_result() -> None:
    """合法 JSON 且引用真实检索来源时，结果可以正常通过。"""

    raw_output = generate_with_mock(
        json.dumps(build_valid_result_payload(), ensure_ascii=False)
    )

    result = parse_service_case_result(
        service_request=build_service_request(),
        raw_model_output=raw_output,
        retrieved_references=[build_retrieved_reference()],
    )

    assert result.classification.event_type.value == "facility_fault"
    assert result.diagnostics.model_output_parse_success is True
    assert result.review.requires_human_review is False


def test_parser_turns_invalid_json_into_human_review_result() -> None:
    """模型返回非 JSON 文本时，系统必须返回可校验的人工复核结果。"""

    result = parse_service_case_result(
        service_request=build_service_request(),
        raw_model_output=generate_with_mock("这不是 JSON"),
        retrieved_references=[],
    )

    assert result.diagnostics.model_output_parse_success is False
    assert result.review.requires_human_review is True
    assert ReviewReason.INVALID_MODEL_OUTPUT in result.review.reasons
    assert ReviewReason.KNOWLEDGE_NOT_FOUND in result.review.reasons
    assert result.model_dump(mode="json")["request_id"] == "demo_001"


def test_knowledge_miss_uses_a_neutral_risk_summary() -> None:
    """知识未命中时，兜底文案不应错误地暗示模型已经返回过内容。"""

    result = build_knowledge_miss_result(build_service_request())

    assert result.risk.summary == "当前无法安全评估风险，需人工复核。"
    assert result.diagnostics.model_call_success is None
    assert result.diagnostics.model_output_parse_success is None


def test_parser_rejects_model_invented_knowledge_reference() -> None:
    """模型即使返回可解析 JSON，也不能伪造本次未检索到的知识来源。"""

    payload = build_valid_result_payload()
    payload["knowledge_references"][0]["source_id"] = "invented_source_001"  # type: ignore[index]
    payload["action_plan"][0]["knowledge_source_ids"] = [  # type: ignore[index]
        "invented_source_001"
    ]

    result = parse_service_case_result(
        service_request=build_service_request(),
        raw_model_output=generate_with_mock(json.dumps(payload, ensure_ascii=False)),
        retrieved_references=[build_retrieved_reference()],
    )

    assert result.diagnostics.model_output_parse_success is False
    assert result.review.requires_human_review is True
    assert ReviewReason.INVALID_MODEL_OUTPUT in result.review.reasons


def test_parser_keeps_valid_high_risk_result_for_human_review() -> None:
    """高风险结果即使结构正确，也必须保留人工复核标记。"""

    payload = build_valid_result_payload()
    payload["risk"] = {
        "level": "high",
        "risk_factors": ["文本包含演示性安全风险描述"],
        "summary": "演示性高风险案件。",
    }
    payload["review"] = {
        "requires_human_review": True,
        "reasons": ["high_risk"],
        "review_note": "高风险必须人工复核。",
    }

    result = parse_service_case_result(
        service_request=build_service_request(),
        raw_model_output=generate_with_mock(json.dumps(payload, ensure_ascii=False)),
        retrieved_references=[build_retrieved_reference()],
    )

    assert result.risk.level == RiskLevel.HIGH
    assert result.review.requires_human_review is True
    assert ReviewReason.HIGH_RISK in result.review.reasons
