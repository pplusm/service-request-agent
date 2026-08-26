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


def build_knowledge_miss_result(
    service_request: ServiceRequestInput,
    retrieval_error: str | None = None,
) -> ServiceCaseResult:
    """在知识未命中或检索失败时生成无需调用模型的人工复核结果。"""

    errors: list[str] = []
    if retrieval_error is not None:
        errors.append(f"knowledge_retrieval_failed: {retrieval_error}"[:500])

    # 没有调用模型时，两个模型状态均使用 None，而不是虚构解析失败。
    return _build_conservative_review_result(
        service_request=service_request,
        retrieved_references=[],
        model_call_success=None,
        model_output_parse_success=None,
        raw_model_output=None,
        errors=errors,
        additional_reasons=[],
        review_note="没有可用的演示知识来源，已转人工复核。",
    )


def build_provider_error_result(
    service_request: ServiceRequestInput,
    retrieved_references: list[KnowledgeReference],
    provider_error: str,
) -> ServiceCaseResult:
    """在模型服务调用失败时生成可校验的人工复核结果。"""

    return _build_conservative_review_result(
        service_request=service_request,
        retrieved_references=retrieved_references,
        model_call_success=False,
        model_output_parse_success=None,
        raw_model_output=None,
        errors=[f"llm_provider_error: {provider_error}"[:500]],
        additional_reasons=[ReviewReason.LLM_ERROR],
        review_note="模型服务调用失败，已转人工复核。",
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

    # Diagnostics 的原始输出字段最长 5,000 个字符，截断可避免二次校验失败。
    safe_raw_output = raw_model_output[:5_000]
    safe_error = f"model_output_validation_failed: {error}"[:500]

    return _build_conservative_review_result(
        service_request=service_request,
        retrieved_references=retrieved_references,
        model_call_success=True,
        model_output_parse_success=False,
        raw_model_output=safe_raw_output,
        errors=[safe_error],
        additional_reasons=[],
        review_note="模型输出无法作为案件结论，已转人工复核。",
    )


def _build_conservative_review_result(
    *,
    service_request: ServiceRequestInput,
    retrieved_references: list[KnowledgeReference],
    model_call_success: bool | None,
    model_output_parse_success: bool | None,
    raw_model_output: str | None,
    errors: list[str],
    additional_reasons: list[ReviewReason],
    review_note: str,
) -> ServiceCaseResult:
    """创建保守的统一兜底结果，集中维护所有人工复核原因。"""

    review_reasons = [
        ReviewReason.UNASSESSED_RISK,
        ReviewReason.MISSING_FIELDS,
        ReviewReason.LOW_CONFIDENCE,
    ]
    if not retrieved_references:
        review_reasons.append(ReviewReason.KNOWLEDGE_NOT_FOUND)
    if model_output_parse_success is False:
        review_reasons.append(ReviewReason.INVALID_MODEL_OUTPUT)
    for reason in additional_reasons:
        if reason not in review_reasons:
            review_reasons.append(reason)

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
            # 该兜底函数也用于知识未命中和模型调用失败，不能一概归因于模型输出。
            summary="当前无法安全评估风险，需人工复核。",
        ),
        # 已检索到的资料仍保留给人工查看，但不会据此生成自动动作建议。
        knowledge_references=retrieved_references,
        action_plan=[],
        review=ReviewDecision(
            requires_human_review=True,
            reasons=review_reasons,
            review_note=review_note,
        ),
        diagnostics=Diagnostics(
            knowledge_hit=bool(retrieved_references),
            model_call_success=model_call_success,
            model_output_parse_success=model_output_parse_success,
            raw_model_output=raw_model_output,
            errors=errors,
        ),
    )
