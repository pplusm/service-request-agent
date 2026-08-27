"""协调 FastAPI 输入校验与 LangGraph Agent 调用。"""

from typing import Any

from app.agent.result_parser import build_input_validation_result
from app.agent.triage_graph import TriageAgent
from app.api.schemas import TriageApiRequest
from app.schemas.models import ServiceCaseResult, ServiceRequestInput


_FALLBACK_REQUEST_ID = "api_invalid_request"


class TriageApiService:
    """将 HTTP 请求转换为安全的 Agent 调用，不让输入校验绕过人工复核。"""

    def __init__(self, agent: TriageAgent) -> None:
        self._agent = agent

    def triage(self, request: TriageApiRequest | None) -> ServiceCaseResult:
        """处理一份 API 请求；缺失字段时不调用模型或知识库。"""

        if request is None:
            return build_input_validation_result(
                request_id=_FALLBACK_REQUEST_ID,
                missing_fields=["request_body"],
                validation_errors=["request body is required"],
            )

        missing_fields: list[str] = []
        if not request.request_id:
            missing_fields.append("request_id")
        if not request.text:
            missing_fields.append("text")

        if missing_fields:
            return build_input_validation_result(
                request_id=request.request_id or _FALLBACK_REQUEST_ID,
                missing_fields=missing_fields,
                validation_errors=[f"missing or empty field: {field}" for field in missing_fields],
            )

        # 经过上方检查后，两个字段都为非空字符串，可安全构造核心输入模型。
        service_request = ServiceRequestInput(
            request_id=request.request_id,
            text=request.text,
            image=request.image,
        )
        return self._agent.run(service_request)


def build_request_validation_error_result(
    validation_errors: list[dict[str, Any]],
) -> ServiceCaseResult:
    """把 FastAPI 在路由前发现的格式错误也转换成统一案件结果。"""

    messages: list[str] = []
    for error in validation_errors[:20]:
        location = ".".join(str(part) for part in error.get("loc", []))
        message = str(error.get("msg", "invalid request body"))
        messages.append(f"{location}: {message}".strip(": "))

    from app.schemas.models import ReviewReason

    image_validation_failed = any(
        "image" in error.get("loc", []) for error in validation_errors
    )
    return build_input_validation_result(
        request_id=_FALLBACK_REQUEST_ID,
        missing_fields=["request_body"],
        validation_errors=messages,
        additional_reasons=(
            [ReviewReason.INVALID_IMAGE] if image_validation_failed else None
        ),
    )
