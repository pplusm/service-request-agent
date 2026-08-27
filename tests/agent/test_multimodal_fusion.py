"""验证图文融合的演示样例和人工复核安全边界。"""

from __future__ import annotations

import base64

from app.agent.multimodal_fusion import build_multimodal_fusion
from app.agent.result_parser import attach_vision_context
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
    ServiceRequestInput,
    VisionObservation,
    MultimodalConflictField,
    MultimodalFusionStatus,
)


def make_observation(
    description: str,
    *,
    confidence: float = 0.90,
    objects: list[str] | None = None,
    visible_text: list[str] | None = None,
    location_hint: str | None = None,
    facility_hint: str | None = None,
    uncertainty_notes: list[str] | None = None,
    is_demo_observation: bool = False,
) -> VisionObservation:
    """构造已通过 Pydantic 校验的模拟视觉观察，不读取真实图片。"""

    return VisionObservation(
        description=description,
        confidence=confidence,
        objects=objects or [],
        visible_text=visible_text or [],
        location_hint=location_hint,
        facility_hint=facility_hint,
        uncertainty_notes=uncertainty_notes or [],
        is_demo_observation=is_demo_observation,
    )


def make_image_request() -> ServiceRequestInput:
    """构造仅用于测试传输的图片请求，不包含真实照片。"""

    return ServiceRequestInput(
        request_id="fusion_context_001",
        text="东门卫生间没水",
        image={
            "media_type": "image/png",
            "data_base64": base64.b64encode(b"fusion-test-image").decode("ascii"),
            "filename": "fusion-demo.png",
        },
    )


def make_result_with_action() -> ServiceCaseResult:
    """构造带演示建议的正常文本结果，用于验证融合失败时会清空建议。"""

    reference = KnowledgeReference(
        source_id="demo_facility_001",
        source_title="演示设施故障资料",
        source_path="data/scenic_service/knowledge/demo_facility.md",
        excerpt="本资料仅用于演示设施故障的可追溯来源。",
        relevance_score=0.90,
    )
    return ServiceCaseResult(
        request_id="fusion_context_001",
        entities=ExtractedEntities(
            location="东门",
            facility_name="卫生间",
            event_time_description="现在",
        ),
        classification=EventClassification(
            event_type=EventType.FACILITY_FAULT,
            confidence=0.90,
            evidence=["东门卫生间没水"],
        ),
        risk=RiskAssessment(
            level=RiskLevel.LOW,
            summary="演示性低风险设施故障。",
        ),
        knowledge_references=[reference],
        action_plan=[
            ActionPlanItem(
                step=1,
                suggested_action="创建演示性的设施维护跟进记录。",
                knowledge_source_ids=[reference.source_id],
            )
        ],
        review=ReviewDecision(requires_human_review=False),
        diagnostics=Diagnostics(
            knowledge_hit=True,
            model_call_success=True,
            model_output_parse_success=True,
        ),
    )


def test_fusion_01_consistent_facility_and_condition() -> None:
    """样例 1：图文都指向东门卫生间无水，应标记为一致。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("东门卫生间的水龙头无水。"),
    )

    assert fusion.status == MultimodalFusionStatus.CONSISTENT
    assert fusion.conflict_fields == []
    assert "设施：卫生间" in fusion.text_concepts
    assert "状态：无水" in fusion.image_concepts


def test_fusion_02_demo_provider_is_not_assessed() -> None:
    """样例 2：本地 Demo 不识别像素，只能返回未评估。"""

    fusion = build_multimodal_fusion(
        request_text="西门照明故障",
        observation=make_observation(
            "已接收一张演示图片。",
            is_demo_observation=True,
        ),
    )

    assert fusion.status == MultimodalFusionStatus.NOT_ASSESSED
    assert fusion.is_demo_assessment is True


def test_fusion_03_low_confidence_is_insufficient_evidence() -> None:
    """样例 3：视觉置信度低于演示门槛时，不能确认图文一致。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("东门卫生间无水。", confidence=0.59),
    )

    assert fusion.status == MultimodalFusionStatus.INSUFFICIENT_EVIDENCE
    assert "置信度" in fusion.note


def test_fusion_04_uncertainty_note_is_insufficient_evidence() -> None:
    """样例 4：视觉观察声明不确定时，不能据此作一致性结论。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation(
            "疑似东门卫生间无水。",
            uncertainty_notes=["图片角度有限，无法确认设施状态。"],
        ),
    )

    assert fusion.status == MultimodalFusionStatus.INSUFFICIENT_EVIDENCE
    assert "未确认" in fusion.note


def test_fusion_05_text_without_demo_concept_is_insufficient_evidence() -> None:
    """样例 5：文字没有当前词典中的可比对概念时，转为证据不足。"""

    fusion = build_multimodal_fusion(
        request_text="游客说这里不太方便",
        observation=make_observation("东门卫生间无水。"),
    )

    assert fusion.status == MultimodalFusionStatus.INSUFFICIENT_EVIDENCE
    assert fusion.text_concepts == []


def test_fusion_06_image_without_demo_concept_is_insufficient_evidence() -> None:
    """样例 6：图片观察没有可核对线索时，不能支持文字诉求。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("画面中有一段普通步道。"),
    )

    assert fusion.status == MultimodalFusionStatus.INSUFFICIENT_EVIDENCE
    assert fusion.image_concepts == []


def test_fusion_07_location_conflict_requires_confirmation() -> None:
    """样例 7：文字和图片给出不同地点时，应标记地点冲突。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("西门卫生间无水。"),
    )

    assert fusion.status == MultimodalFusionStatus.CONFLICT
    assert fusion.conflict_fields == [MultimodalConflictField.LOCATION]


def test_fusion_08_facility_conflict_requires_confirmation() -> None:
    """样例 8：文字和图片指向不同设施时，应标记设施冲突。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("东门路灯不亮。"),
    )

    assert fusion.status == MultimodalFusionStatus.CONFLICT
    assert fusion.conflict_fields == [MultimodalConflictField.FACILITY]


def test_fusion_09_condition_conflict_requires_confirmation() -> None:
    """样例 9：同一设施状态相反时，应标记状态冲突。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("东门卫生间水龙头正常供水。"),
    )

    assert fusion.status == MultimodalFusionStatus.CONFLICT
    assert fusion.conflict_fields == [MultimodalConflictField.CONDITION]


def test_fusion_10_same_location_without_issue_support_is_insufficient() -> None:
    """样例 10：只有地点相同不足以确认事件，仍需更多设施或状态证据。"""

    fusion = build_multimodal_fusion(
        request_text="东门卫生间没水",
        observation=make_observation("东门入口附近的道路状况正常。"),
    )

    assert fusion.status == MultimodalFusionStatus.INSUFFICIENT_EVIDENCE
    assert "可核对" not in fusion.note


def test_non_consistent_fusion_clears_action_plan_and_requires_review() -> None:
    """图文冲突时，即使文本模型给出建议，也必须清空并转人工复核。"""

    request = make_image_request()
    observation = make_observation("西门卫生间无水。")
    fusion = build_multimodal_fusion(
        request_text=request.text,
        observation=observation,
    )

    result = attach_vision_context(
        result=make_result_with_action(),
        service_request=request,
        observation=observation,
        vision_call_success=True,
        vision_output_parse_success=True,
        raw_vision_output=None,
        vision_provider_name="test-vision-provider",
        vision_model_name="test-vision-model",
        fusion=fusion,
    )

    assert fusion.status == MultimodalFusionStatus.CONFLICT
    assert result.action_plan == []
    assert result.review.requires_human_review is True
    assert ReviewReason.MULTIMODAL_CONFLICT in result.review.reasons
