"""在 LangGraph 终态统一执行可配置的高风险安全升级。"""

from app.rules.scenic_service_config import ScenicServiceConfiguration
from app.schemas.models import ReviewReason, ServiceCaseResult, ServiceRequestInput


def enforce_configured_high_risk_rule(
    *,
    result: ServiceCaseResult,
    service_request: ServiceRequestInput,
    configuration: ScenicServiceConfiguration,
) -> ServiceCaseResult:
    """命中 YAML 高风险词时覆盖模型结论，并清除自动动作建议。"""

    risk_rule = configuration.find_high_risk_rule(service_request.text)
    if risk_rule is None:
        return result

    # 先转成 JSON 兼容字典，再重新使用 Pydantic 校验，避免直接修改嵌套模型绕过契约。
    payload = result.model_dump(mode="json")
    risk_factors = list(payload["risk"]["risk_factors"])
    configured_factor = f"命中演示风险规则：{risk_rule.rule_id}"
    if configured_factor not in risk_factors:
        risk_factors.append(configured_factor)

    payload["risk"] = {
        "level": risk_rule.risk_level.value,
        "risk_factors": risk_factors,
        "summary": risk_rule.risk_summary,
    }

    # 高风险不能保留看似可直接执行的自动建议，必须交由人工确认。
    payload["action_plan"] = []
    review = payload["review"]
    review["requires_human_review"] = True
    if ReviewReason.HIGH_RISK.value not in review["reasons"]:
        review["reasons"].append(ReviewReason.HIGH_RISK.value)

    original_note = str(review["review_note"]).strip()
    notes = [risk_rule.review_note]
    if original_note and original_note != risk_rule.review_note:
        notes.append(original_note)
    review["review_note"] = " ".join(notes)[:500]

    return ServiceCaseResult.model_validate(payload)
