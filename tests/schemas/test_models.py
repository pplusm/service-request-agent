import json

import pytest
from pydantic import ValidationError

from app.schemas.models import (
    ActionPlanItem,
    Diagnostics,
    EventClassification,
    EventType,
    ExtractedEntities,
    KnowledgeReference,
    ReviewDecision,
    ReviewReason,
    RiskAssessment,
    RiskLevel,
    ServiceCaseResult,
)


def make_demo_knowledge_reference() -> KnowledgeReference:
    """创建可重复使用的演示知识来源，不使用真实景区规则。"""

    return KnowledgeReference(
        source_id="demo_facility_001",
        source_title="Demo facility guidance",
        source_path="data/scenic_service/knowledge/demo_facility.md",
        excerpt="Demo source only. Confirm the case before any real action.",
        relevance_score=0.85,
    )


def make_demo_action() -> ActionPlanItem:
    """创建与演示知识来源绑定的建议动作。"""

    return ActionPlanItem(
        step=1,
        suggested_action="Create a demo maintenance follow-up record.",
        knowledge_source_ids=["demo_facility_001"],
    )


def test_demo_001_normal_facility_case_does_not_require_review() -> None:
    """低风险、知识命中的设施故障案例可以通过结构校验。"""

    result = ServiceCaseResult(
        request_id="demo_001_normal_facility",
        entities=ExtractedEntities(
            location="east gate restroom",
            facility_name="restroom",
            event_time_description="now",
        ),
        classification=EventClassification(
            event_type=EventType.FACILITY_FAULT,
            confidence=0.88,
            evidence=["restroom has no water"],
        ),
        risk=RiskAssessment(
            level=RiskLevel.LOW,
            risk_factors=["single facility issue"],
            summary="Demo low-risk facility case.",
        ),
        knowledge_references=[make_demo_knowledge_reference()],
        action_plan=[make_demo_action()],
        review=ReviewDecision(requires_human_review=False),
        diagnostics=Diagnostics(knowledge_hit=True),
    )

    # model_dump_json() 证明最终结果可以序列化为标准 JSON。
    payload = json.loads(result.model_dump_json())
    assert payload["review"]["requires_human_review"] is False
    assert payload["knowledge_references"][0]["is_demo_source"] is True


def test_demo_002_high_risk_requires_human_review() -> None:
    """高风险案例若未开启人工复核，Pydantic 必须拒绝该结果。"""

    with pytest.raises(ValidationError, match="human review is required"):
        ServiceCaseResult(
            request_id="demo_002_high_risk",
            entities=ExtractedEntities(
                location="west trail",
                visitor_condition="possible injury",
                event_time_description="now",
            ),
            classification=EventClassification(
                event_type=EventType.VISITOR_HEALTH,
                confidence=0.95,
                evidence=["visitor reports an injury"],
            ),
            risk=RiskAssessment(
                level=RiskLevel.HIGH,
                risk_factors=["possible injury"],
                summary="Demo high-risk case.",
            ),
            knowledge_references=[make_demo_knowledge_reference()],
            action_plan=[make_demo_action()],
            review=ReviewDecision(requires_human_review=False),
            diagnostics=Diagnostics(knowledge_hit=True),
        )

    # 补齐人工复核开关和原因后，同一个高风险案例才可以通过校验。
    valid_result = ServiceCaseResult(
        request_id="demo_002_high_risk",
        entities=ExtractedEntities(
            location="west trail",
            visitor_condition="possible injury",
            event_time_description="now",
        ),
        classification=EventClassification(
            event_type=EventType.VISITOR_HEALTH,
            confidence=0.95,
            evidence=["visitor reports an injury"],
        ),
        risk=RiskAssessment(
            level=RiskLevel.HIGH,
            risk_factors=["possible injury"],
            summary="Demo high-risk case.",
        ),
        knowledge_references=[make_demo_knowledge_reference()],
        action_plan=[make_demo_action()],
        review=ReviewDecision(
            requires_human_review=True,
            reasons=[ReviewReason.HIGH_RISK],
        ),
        diagnostics=Diagnostics(knowledge_hit=True),
    )

    assert valid_result.review.requires_human_review is True


def test_demo_003_missing_fields_requires_human_review() -> None:
    """关键信息缺失时，结果必须进入人工复核。"""

    result = ServiceCaseResult(
        request_id="demo_003_missing_fields",
        entities=ExtractedEntities(
            missing_fields=["location", "event_time_description"],
        ),
        classification=EventClassification(
            event_type=EventType.OTHER_UNKNOWN,
            confidence=0.20,
            evidence=["vague request text"],
        ),
        risk=RiskAssessment(
            level=RiskLevel.UNASSESSED,
            summary="Insufficient information in demo request.",
        ),
        review=ReviewDecision(
            requires_human_review=True,
            reasons=[
                ReviewReason.UNASSESSED_RISK,
                ReviewReason.MISSING_FIELDS,
                ReviewReason.KNOWLEDGE_NOT_FOUND,
                ReviewReason.LOW_CONFIDENCE,
            ],
        ),
        diagnostics=Diagnostics(knowledge_hit=False),
    )

    assert ReviewReason.MISSING_FIELDS in result.review.reasons


def test_demo_004_knowledge_miss_requires_review_and_no_action() -> None:
    """知识库未命中时必须人工复核，且不能保留处置建议。"""

    result = ServiceCaseResult(
        request_id="demo_004_knowledge_miss",
        entities=ExtractedEntities(
            location="north gate",
            event_time_description="now",
        ),
        classification=EventClassification(
            event_type=EventType.OTHER_UNKNOWN,
            confidence=0.75,
            evidence=["request is outside demo knowledge"],
        ),
        risk=RiskAssessment(
            level=RiskLevel.MEDIUM,
            summary="Demo request without a knowledge match.",
        ),
        review=ReviewDecision(
            requires_human_review=True,
            reasons=[ReviewReason.KNOWLEDGE_NOT_FOUND],
        ),
        diagnostics=Diagnostics(knowledge_hit=False),
    )

    assert result.knowledge_references == []
    assert result.action_plan == []
    assert result.review.requires_human_review is True


def test_demo_005_invalid_model_output_returns_valid_review_result() -> None:
    """模型输出无法解析时，保留原始内容并生成可校验的人工复核结果。"""

    raw_model_output = "{event_type: facility_fault, confidence: not-a-number}"

    fallback_result = ServiceCaseResult(
        request_id="demo_005_invalid_model_output",
        entities=ExtractedEntities(
            missing_fields=["location", "event_time_description"],
        ),
        classification=EventClassification(
            event_type=EventType.OTHER_UNKNOWN,
            confidence=0.0,
        ),
        risk=RiskAssessment(
            level=RiskLevel.UNASSESSED,
            summary="Fallback result after demo parsing failure.",
        ),
        review=ReviewDecision(
            requires_human_review=True,
            reasons=[
                ReviewReason.UNASSESSED_RISK,
                ReviewReason.MISSING_FIELDS,
                ReviewReason.KNOWLEDGE_NOT_FOUND,
                ReviewReason.INVALID_MODEL_OUTPUT,
                ReviewReason.LOW_CONFIDENCE,
            ],
        ),
        diagnostics=Diagnostics(
            knowledge_hit=False,
            model_output_parse_success=False,
            raw_model_output=raw_model_output,
            errors=["Demo model output could not be parsed as JSON."],
        ),
    )

    # 即使模型原始输出错误，fallback_result 自身仍是可序列化的有效 JSON。
    payload = json.loads(fallback_result.model_dump_json())
    assert payload["diagnostics"]["raw_model_output"] == raw_model_output
    assert payload["review"]["requires_human_review"] is True