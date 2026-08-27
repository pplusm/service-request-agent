"""将模型原始输出转换为经过 Pydantic 校验的案件结果。"""

import re

from pydantic import ValidationError

from app.schemas.models import (
    Diagnostics,
    EventClassification,
    EventType,
    ExtractedEntities,
    KnowledgeReference,
    ImageMetadata,
    MultimodalFusion,
    MultimodalFusionStatus,
    ReviewDecision,
    ReviewReason,
    RiskAssessment,
    RiskLevel,
    ServiceCaseResult,
    ServiceRequestInput,
    VisionObservation,
)


# 有些第三方服务会在错误信息中回显请求体；这条规则用于删除其中的图片 data URL。
_IMAGE_DATA_URL_PATTERN = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]+",
    re.IGNORECASE,
)


def redact_image_payload_from_diagnostics(
    value: str | None,
    service_request: ServiceRequestInput,
    *,
    max_length: int,
) -> str | None:
    """从诊断文本中移除图片内容，再按字段上限截断。

    图片 Base64 仅允许在本次视觉调用的内存请求中存在。视觉提供方或异常对象
    若把它回显到原始输出、错误信息或调用链异常中，返回 JSON 和本地历史都不能
    保存这部分内容。
    """

    if value is None:
        return None

    # 先匹配完整 data URL，避免只替换其中的 base64 后残留可误导使用者的 URL 前缀。
    redacted = _IMAGE_DATA_URL_PATTERN.sub("[REDACTED_IMAGE_DATA_URL]", value)
    if service_request.image is not None:
        # 再处理服务方可能单独回显的原始 base64，不依赖具体图片格式。
        redacted = redacted.replace(
            service_request.image.data_base64,
            "[REDACTED_IMAGE_BASE64]",
        )
    return redacted[:max_length]


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
        errors.append(f"knowledge_retrieval_failed: {retrieval_error}")

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


def build_input_validation_result(
    *,
    request_id: str,
    missing_fields: list[str],
    validation_errors: list[str],
    additional_reasons: list[ReviewReason] | None = None,
) -> ServiceCaseResult:
    """将 API 输入字段缺失或格式无效转为统一的人工复核结果。"""

    # ServiceCaseResult 不返回原始文本；这里的占位文本只用于复用统一兜底构造逻辑。
    placeholder_request = ServiceRequestInput(
        request_id=request_id,
        text="输入字段校验失败",
    )
    normalized_fields = [
        field.strip()[:100] for field in missing_fields if field.strip()
    ]
    if not normalized_fields:
        normalized_fields = ["request_body"]

    errors = [
        f"input_validation_failed: {error}"[:500]
        for error in validation_errors[:20]
    ]
    if not errors:
        errors = ["input_validation_failed: unknown request validation error"]

    return _build_conservative_review_result(
        service_request=placeholder_request,
        retrieved_references=[],
        model_call_success=None,
        model_output_parse_success=None,
        raw_model_output=None,
        errors=errors,
        additional_reasons=[
            ReviewReason.MISSING_FIELDS,
            *(additional_reasons or []),
        ],
        review_note="请求字段缺失或格式无效，已转人工复核。",
        missing_fields=normalized_fields,
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
        errors=[f"llm_provider_error: {provider_error}"],
        additional_reasons=[ReviewReason.LLM_ERROR],
        review_note="模型服务调用失败，已转人工复核。",
    )


def build_vision_error_result(
    *,
    service_request: ServiceRequestInput,
    retrieved_references: list[KnowledgeReference],
    vision_output_parse_success: bool | None,
    raw_vision_output: str | None,
    vision_error: str,
    vision_call_success: bool,
    vision_provider_name: str | None = None,
    vision_model_name: str | None = None,
) -> ServiceCaseResult:
    """在视觉调用或视觉 JSON 解析失败时生成保守的人工复核结果。"""

    if vision_output_parse_success is False:
        reason = ReviewReason.INVALID_VISION_OUTPUT
        review_note = "视觉模型输出无法校验，已转人工复核。"
    else:
        reason = ReviewReason.VISION_ERROR
        review_note = "视觉模型调用失败，已转人工复核。"

    return _build_conservative_review_result(
        service_request=service_request,
        retrieved_references=retrieved_references,
        model_call_success=None,
        model_output_parse_success=None,
        raw_model_output=None,
        errors=[f"vision_provider_error: {vision_error}"],
        additional_reasons=[reason],
        review_note=review_note,
        image_metadata=(
            service_request.image.metadata()
            if service_request.image is not None
            else None
        ),
        vision_observation=None,
        vision_call_success=vision_call_success,
        vision_output_parse_success=vision_output_parse_success,
        raw_vision_output=raw_vision_output,
        vision_provider_name=vision_provider_name,
        vision_model_name=vision_model_name,
    )


def attach_vision_context(
    *,
    result: ServiceCaseResult,
    service_request: ServiceRequestInput,
    observation: VisionObservation | None,
    vision_call_success: bool | None,
    vision_output_parse_success: bool | None,
    raw_vision_output: str | None,
    vision_provider_name: str | None,
    vision_model_name: str | None,
    fusion: MultimodalFusion | None = None,
    failure_reason: ReviewReason | None = None,
) -> ServiceCaseResult:
    """把视觉节点和图文融合的可信摘要合并到最终结果中。"""

    if service_request.image is None:
        # 没有图片时不应凭空写入任何视觉字段，保持文本 MVP 的 JSON 形状。
        return result

    payload = result.model_dump(mode="json")
    payload["image"] = service_request.image.metadata().model_dump(mode="json")
    payload["vision_observation"] = (
        observation.model_dump(mode="json") if observation is not None else None
    )
    payload["multimodal_fusion"] = (
        fusion.model_dump(mode="json") if fusion is not None else None
    )
    diagnostics = payload["diagnostics"]
    diagnostics.update(
        {
            "vision_call_success": vision_call_success,
            "vision_output_parse_success": vision_output_parse_success,
            "raw_vision_output": redact_image_payload_from_diagnostics(
                raw_vision_output,
                service_request,
                max_length=5_000,
            ),
            "vision_provider_name": vision_provider_name,
            "vision_model_name": vision_model_name,
        }
    )

    review = payload["review"]
    if failure_reason is not None:
        review["requires_human_review"] = True
        reasons = review["reasons"]
        if failure_reason.value not in reasons:
            reasons.append(failure_reason.value)

    if fusion is not None and fusion.status != MultimodalFusionStatus.CONSISTENT:
        # 图文冲突、视觉信息不足和本地 demo 的“未评估”都不能保留自动建议。
        review["requires_human_review"] = True
        if fusion.status == MultimodalFusionStatus.CONFLICT:
            fusion_reason = ReviewReason.MULTIMODAL_CONFLICT
        else:
            fusion_reason = ReviewReason.MULTIMODAL_INSUFFICIENT_EVIDENCE
        if fusion_reason.value not in review["reasons"]:
            review["reasons"].append(fusion_reason.value)
        payload["action_plan"] = []

        # 原有复核说明和融合说明都保留，便于页面和历史记录解释为何不能自动处置。
        original_note = str(review["review_note"]).strip()
        fusion_note = f"图文融合：{fusion.note}"
        review["review_note"] = " ".join(
            note for note in (original_note, fusion_note) if note
        )[:500]

    return ServiceCaseResult.model_validate(payload)


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

    # 图片摘要和视觉观察由 Agent 节点生成，模型不能自行伪造或覆盖它们。
    if result.image is not None or result.vision_observation is not None:
        raise ValueError("model result must not contain vision context")

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

    # 脱敏和截断由统一兜底函数完成，避免先截断后留下图片 Base64 的前半段。
    safe_raw_output = raw_model_output
    safe_error = f"model_output_validation_failed: {error}"

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
    missing_fields: list[str] | None = None,
    image_metadata: ImageMetadata | None = None,
    vision_observation: VisionObservation | None = None,
    vision_call_success: bool | None = None,
    vision_output_parse_success: bool | None = None,
    raw_vision_output: str | None = None,
    vision_provider_name: str | None = None,
    vision_model_name: str | None = None,
) -> ServiceCaseResult:
    """创建保守的统一兜底结果，集中维护所有人工复核原因。"""

    # 所有兜底路径都经过这里，因此集中处理视觉提供方可能回显的图片内容。
    safe_raw_model_output = redact_image_payload_from_diagnostics(
        raw_model_output,
        service_request,
        max_length=5_000,
    )
    safe_raw_vision_output = redact_image_payload_from_diagnostics(
        raw_vision_output,
        service_request,
        max_length=5_000,
    )
    safe_errors = [
        safe_error
        for error in errors
        if (
            safe_error := redact_image_payload_from_diagnostics(
                error,
                service_request,
                max_length=500,
            )
        ) is not None
    ]

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
            missing_fields=missing_fields
            if missing_fields is not None
            else ["location", "event_time_description"]
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
        image=image_metadata,
        vision_observation=vision_observation,
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
            raw_model_output=safe_raw_model_output,
            vision_call_success=vision_call_success,
            vision_output_parse_success=vision_output_parse_success,
            raw_vision_output=safe_raw_vision_output,
            vision_provider_name=vision_provider_name,
            vision_model_name=vision_model_name,
            errors=safe_errors,
        ),
    )
