"""用确定性、可测试的规则融合景区文本与结构化视觉观察。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.schemas.models import (
    MultimodalConflictField,
    MultimodalFusion,
    MultimodalFusionStatus,
    VisionObservation,
)


# 这是项目演示使用的保守门槛，不代表任何视觉模型的真实准确率。
_MIN_CONFIDENT_VISION_SCORE: Final = 0.60


@dataclass(frozen=True)
class _ConceptRule:
    """一个可由文字或视觉观察匹配到的受控演示概念。"""

    field: MultimodalConflictField
    identifier: str
    label: str
    aliases: tuple[str, ...]
    condition_group: str | None = None
    condition_state: str | None = None


# 地点、设施和状态词只覆盖当前景区 Demo 的有限范围，不构成通用识别词典。
_CONCEPT_RULES: Final = (
    _ConceptRule(
        MultimodalConflictField.LOCATION,
        "east_gate",
        "地点：东门",
        ("东门",),
    ),
    _ConceptRule(
        MultimodalConflictField.LOCATION,
        "west_gate",
        "地点：西门",
        ("西门",),
    ),
    _ConceptRule(
        MultimodalConflictField.LOCATION,
        "south_gate",
        "地点：南门",
        ("南门",),
    ),
    _ConceptRule(
        MultimodalConflictField.LOCATION,
        "north_gate",
        "地点：北门",
        ("北门",),
    ),
    _ConceptRule(
        MultimodalConflictField.LOCATION,
        "visitor_center",
        "地点：游客中心",
        ("游客中心",),
    ),
    _ConceptRule(
        MultimodalConflictField.LOCATION,
        "parking_lot",
        "地点：停车场",
        ("停车场",),
    ),
    _ConceptRule(
        MultimodalConflictField.FACILITY,
        "restroom",
        "设施：卫生间",
        ("卫生间", "洗手间", "厕所"),
    ),
    _ConceptRule(
        MultimodalConflictField.FACILITY,
        "lighting",
        "设施：照明设施",
        ("照明设施", "照明", "路灯", "灯具", "灯光"),
    ),
    _ConceptRule(
        MultimodalConflictField.FACILITY,
        "signage",
        "设施：指示牌",
        ("指示牌", "导览牌", "标识牌"),
    ),
    _ConceptRule(
        MultimodalConflictField.CONDITION,
        "water_unavailable",
        "状态：无水",
        ("没水", "无水", "停水", "没有水", "水龙头无水"),
        condition_group="water",
        condition_state="fault",
    ),
    _ConceptRule(
        MultimodalConflictField.CONDITION,
        "water_available",
        "状态：供水正常",
        ("正常供水", "水龙头正常", "有水"),
        condition_group="water",
        condition_state="normal",
    ),
    _ConceptRule(
        MultimodalConflictField.CONDITION,
        "lighting_fault",
        "状态：照明故障",
        (
            "照明故障",
            "路灯故障",
            "照明不亮",
            "路灯不亮",
            "灯光不亮",
            "照明损坏",
            "路灯损坏",
        ),
        condition_group="lighting",
        condition_state="fault",
    ),
    _ConceptRule(
        MultimodalConflictField.CONDITION,
        "lighting_normal",
        "状态：照明正常",
        ("照明正常", "路灯正常", "灯光正常", "照明完好", "路灯完好"),
        condition_group="lighting",
        condition_state="normal",
    ),
    _ConceptRule(
        MultimodalConflictField.CONDITION,
        "signage_damaged",
        "状态：指示牌损坏",
        ("指示牌损坏", "指示牌破损", "指示牌坏了"),
        condition_group="signage",
        condition_state="fault",
    ),
    _ConceptRule(
        MultimodalConflictField.CONDITION,
        "signage_normal",
        "状态：指示牌正常",
        ("指示牌正常", "指示牌完好"),
        condition_group="signage",
        condition_state="normal",
    ),
)


@dataclass(frozen=True)
class _MatchedConcept:
    """保留命中的规则，便于在比较时读取字段、状态和展示标签。"""

    rule: _ConceptRule


def build_multimodal_fusion(
    *,
    request_text: str,
    observation: VisionObservation,
) -> MultimodalFusion:
    """根据文本和视觉观察生成保守的图文融合结论。

    本函数只比较固定的演示概念。未识别到概念、视觉置信度不足、存在不确定性，
    或当前使用本地 demo 视觉模型时，都不会伪造“图文一致”的结论。
    """

    text_concepts = _extract_concepts(request_text)

    # Demo 视觉提供方不会读取真实像素，因此它不能参与真实图文比对。
    if observation.is_demo_observation:
        return MultimodalFusion(
            status=MultimodalFusionStatus.NOT_ASSESSED,
            text_concepts=_labels(text_concepts),
            image_concepts=[],
            note="本地演示视觉模型未分析图片像素，无法核对图文信息。",
            is_demo_assessment=True,
        )

    image_concepts = _extract_concepts(_observation_evidence_text(observation))

    if observation.confidence < _MIN_CONFIDENT_VISION_SCORE:
        return _build_insufficient_evidence(
            text_concepts=text_concepts,
            image_concepts=image_concepts,
            note=(
                "视觉观察置信度低于演示融合门槛，不能据此确认图文一致性。"
            ),
        )

    if observation.uncertainty_notes:
        return _build_insufficient_evidence(
            text_concepts=text_concepts,
            image_concepts=image_concepts,
            note="视觉观察含有未确认内容，不能据此确认图文一致性。",
        )

    if not text_concepts:
        return _build_insufficient_evidence(
            text_concepts=text_concepts,
            image_concepts=image_concepts,
            note="文本未命中可核对的演示概念，不能完成图文一致性判断。",
        )

    if not image_concepts:
        return _build_insufficient_evidence(
            text_concepts=text_concepts,
            image_concepts=image_concepts,
            note="图片观察未提供可核对的演示概念，不能确认文字诉求。",
        )

    conflict_fields = _find_conflict_fields(text_concepts, image_concepts)
    if conflict_fields:
        return MultimodalFusion(
            status=MultimodalFusionStatus.CONFLICT,
            text_concepts=_labels(text_concepts),
            image_concepts=_labels(image_concepts),
            conflict_fields=conflict_fields,
            note="图文中的地点、设施或状态线索存在冲突，已保留给人工确认。",
        )

    if _has_issue_support(text_concepts, image_concepts):
        return MultimodalFusion(
            status=MultimodalFusionStatus.CONSISTENT,
            text_concepts=_labels(text_concepts),
            image_concepts=_labels(image_concepts),
            note="图片观察在当前演示概念范围内支持文字诉求。",
        )

    return _build_insufficient_evidence(
        text_concepts=text_concepts,
        image_concepts=image_concepts,
        note="图文虽未发现直接冲突，但缺少能够支持同一设施或状态的证据。",
    )


def _build_insufficient_evidence(
    *,
    text_concepts: list[_MatchedConcept],
    image_concepts: list[_MatchedConcept],
    note: str,
) -> MultimodalFusion:
    """统一生成“证据不足”的融合结果，避免各分支遗漏字段。"""

    return MultimodalFusion(
        status=MultimodalFusionStatus.INSUFFICIENT_EVIDENCE,
        text_concepts=_labels(text_concepts),
        image_concepts=_labels(image_concepts),
        note=note,
    )


def _observation_evidence_text(observation: VisionObservation) -> str:
    """只拼接模型已声明为可见的内容，不把不确定性说明当作观察事实。"""

    evidence_parts = [
        observation.description,
        *observation.objects,
        *observation.visible_text,
        observation.location_hint or "",
        observation.facility_hint or "",
        *observation.hazard_signals,
    ]
    return "\n".join(part for part in evidence_parts if part.strip())


def _extract_concepts(text: str) -> list[_MatchedConcept]:
    """从一段文本中提取有限的受控概念，保留规则定义的固定顺序。"""

    normalized_text = text.lower().strip()
    matched: list[_MatchedConcept] = []
    for rule in _CONCEPT_RULES:
        if _rule_matches(rule, normalized_text):
            matched.append(_MatchedConcept(rule=rule))
    return matched


def _rule_matches(rule: _ConceptRule, normalized_text: str) -> bool:
    """处理普通别名匹配，并避免“没有水”被误识别为“有水”。"""

    if not normalized_text:
        return False
    if rule.identifier == "water_available" and any(
        negative_alias in normalized_text
        for negative_alias in ("没水", "无水", "停水", "没有水", "水龙头无水")
    ):
        return False
    return any(alias in normalized_text for alias in rule.aliases)


def _labels(concepts: list[_MatchedConcept]) -> list[str]:
    """把内部概念转换成对用户可读、但不含原始文本的固定标签。"""

    return [concept.rule.label for concept in concepts]


def _find_conflict_fields(
    text_concepts: list[_MatchedConcept],
    image_concepts: list[_MatchedConcept],
) -> list[MultimodalConflictField]:
    """只在双方都给出同一维度且明确相反时标记冲突。"""

    conflicts: list[MultimodalConflictField] = []
    for field in (
        MultimodalConflictField.LOCATION,
        MultimodalConflictField.FACILITY,
    ):
        text_identifiers = _identifiers_for_field(text_concepts, field)
        image_identifiers = _identifiers_for_field(image_concepts, field)
        if (
            text_identifiers
            and image_identifiers
            and text_identifiers.isdisjoint(image_identifiers)
        ):
            conflicts.append(field)

    if _has_condition_conflict(text_concepts, image_concepts):
        conflicts.append(MultimodalConflictField.CONDITION)
    return conflicts


def _identifiers_for_field(
    concepts: list[_MatchedConcept],
    field: MultimodalConflictField,
) -> set[str]:
    """读取某一维度中已经识别出的概念编号。"""

    return {
        concept.rule.identifier
        for concept in concepts
        if concept.rule.field == field
    }


def _has_condition_conflict(
    text_concepts: list[_MatchedConcept],
    image_concepts: list[_MatchedConcept],
) -> bool:
    """状态只有在同一设施组出现“故障”和“正常”时才算冲突。"""

    text_states = _condition_states(text_concepts)
    image_states = _condition_states(image_concepts)
    for group, text_state in text_states.items():
        image_state = image_states.get(group)
        if image_state is not None and image_state != text_state:
            return True
    return False


def _condition_states(concepts: list[_MatchedConcept]) -> dict[str, str]:
    """把状态概念按设施组归并为“fault”或“normal”。"""

    return {
        concept.rule.condition_group: concept.rule.condition_state
        for concept in concepts
        if (
            concept.rule.field == MultimodalConflictField.CONDITION
            and concept.rule.condition_group is not None
            and concept.rule.condition_state is not None
        )
    }


def _has_issue_support(
    text_concepts: list[_MatchedConcept],
    image_concepts: list[_MatchedConcept],
) -> bool:
    """确认图片至少支持同一设施或同一状态，地点一致本身不足以确认事件。"""

    text_issue_identifiers = {
        concept.rule.identifier
        for concept in text_concepts
        if concept.rule.field
        in {MultimodalConflictField.FACILITY, MultimodalConflictField.CONDITION}
    }
    image_issue_identifiers = {
        concept.rule.identifier
        for concept in image_concepts
        if concept.rule.field
        in {MultimodalConflictField.FACILITY, MultimodalConflictField.CONDITION}
    }
    return bool(text_issue_identifiers & image_issue_identifiers)
