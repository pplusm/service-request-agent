"""将模型原始输出转换为经过 Pydantic 校验的案件结果。"""

from pydantic import ValidationError

from app.schemas.models import (
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
)


def parse_service_case_result(
    service_request: ServiceRequestInput,
    raw_model_output: str,
    retrieved_references: list[KnowledgeReference],
) -> ServiceCaseResult:
    """解析模型 JSON；任何不安全或无效的结果都会转为人工复核。"""

    try:
        result = ServiceCaseResult.model_validate_json(raw_model_output)
        _validate_result_matches_request(
            result=result,
            service_request=service_request,
            retrieved_references=retrieved_references,
        )
        return result
    except (ValidationError, ValueError) as error:
        return _build_parse_failure_result(
            service_request=service_request,
            raw_model_output=raw_model_output,
            retrieved_references=retrieved_references,
            error=error,
        )


def _validate_result_matches_request(
    *,
    result: ServiceCaseResult,
    service_request: ServiceRequestInput,
    retrieved_references: list[KnowledgeReference],
) -> None:
    """确认模型没有把其他案件或未检索到的资料混入本次结果。"""

    if result.request_id != service_request.request_id:
        raise ValueError("model result request_id does not match input request_id")

    if result.scenario != service_request.scenario:
        raise ValueError("model result scenario does not match input scenario")

    retrieved_by_id = {
        reference.source_id: reference for reference in retrieved_references
    }
    if len(retrieved_by_id) != len(retrieved_references):
        raise ValueError("retrieved references contain duplicate source ids")

    # 模型引用的每份资料都必须与本次 RAG 实际返回的资料完全一致。
    for reference in result.knowledge_references:
        retrieved_reference = retrieved_by_id.get(reference.source_id)
        if retrieved_reference != reference:
            raise ValueError(
                "model result contains a knowledge reference that was not retrieved"
            )


def _build_parse_failure_result(
    *,
    service_request: ServiceRequestInput,
    raw_model_output: str,
    retrieved_references: list[KnowledgeReference],
    error: Exception,
) -> ServiceCaseResult:
    """构造保守的人工复核结果，确保失败时仍能返回合法 JSON。"""

    review_reasons = [
        ReviewReason.UNASSESSED_RISK,
        ReviewReason.MISSING_FIELDS,
        ReviewReason.INVALID_MODEL_OUTPUT,
        ReviewReason.LOW_CONFIDENCE,
    ]
    if not retrieved_references:
        review_reasons.append(ReviewReason.KNOWLEDGE_NOT_FOUND)

    # Diagnostics 的原始输出字段最长 5,000 个字符，截断可避免二次校验失败。
    safe_raw_output = raw_model_output[:5_000]
    safe_error = f"model_output_validation_failed: {error}"[:500]

    return ServiceCaseResult(
        request_id=service_request.request_id,
        scenario=service_request.scenario,
        entities=ExtractedEntities(
            missing_fields=["location", "event_time_description"]
        ),
        classification=EventClassification(
            event_type=EventType.OTHER_UNKNOWN,
            confidence=0.0,
            evidence=[],
        ),
        risk=RiskAssessment(
            level=RiskLevel.UNASSESSED,
            risk_factors=[],
            summary="模型输出未能安全校验，风险暂不评估。",
        ),
        # 已检索到的资料仍保留给人工查看，但不会据此生成自动动作建议。
        knowledge_references=retrieved_references,
        action_plan=[],
        review=ReviewDecision(
            requires_human_review=True,
            reasons=review_reasons,
            review_note="模型输出无法作为案件结论，已转人工复核。",
        ),
        diagnostics=Diagnostics(
            knowledge_hit=bool(retrieved_references),
            model_output_parse_success=False,
            raw_model_output=safe_raw_output,
            errors=[safe_error],
        ),
    )
