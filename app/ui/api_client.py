"""供 Streamlit 页面调用 FastAPI 分诊接口的客户端。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.case_history.models import CaseHistoryResponse, HumanReviewQueueResponse
from app.schemas.models import ServiceCaseResult


class TriageApiClientError(RuntimeError):
    """页面调用 API 时发生的可展示错误。"""


class TriageApiConnectionError(TriageApiClientError):
    """本地 FastAPI 服务不可连接时使用。"""


class TriageApiResponseError(TriageApiClientError):
    """API 返回的内容无法通过核心 Pydantic 模型校验时使用。"""


def submit_triage_request(
    *,
    api_base_url: str,
    request_id: str,
    text: str,
    timeout_seconds: float = 15.0,
) -> ServiceCaseResult:
    """提交文本诉求，并只返回经过 Pydantic 校验的案件结果。

    即使 FastAPI 因输入问题返回 422，本项目的 API 也会给出符合
    ``ServiceCaseResult`` 的人工复核结果，因此仍按正常结果解析。
    """

    request_payload = json.dumps(
        {"request_id": request_id, "text": text},
        ensure_ascii=False,
    ).encode("utf-8")
    http_request = Request(
        url=_build_api_url(api_base_url, "/api/v1/triage"),
        data=request_payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    return _parse_service_case_result(
        _send_api_request(http_request, timeout_seconds=timeout_seconds)
    )


def fetch_case_history(
    *,
    api_base_url: str,
    timeout_seconds: float = 15.0,
) -> CaseHistoryResponse:
    """读取最近保存的本地案件历史，并拒绝未通过 Pydantic 校验的响应。"""

    http_request = Request(
        url=_build_api_url(api_base_url, "/api/v1/case-history"),
        method="GET",
    )
    return _parse_case_history_response(
        _send_api_request(http_request, timeout_seconds=timeout_seconds)
    )


def fetch_human_review_queue(
    *,
    api_base_url: str,
    timeout_seconds: float = 15.0,
) -> HumanReviewQueueResponse:
    """读取待人工复核列表，并拒绝未通过 Pydantic 校验的响应。"""

    http_request = Request(
        url=_build_api_url(api_base_url, "/api/v1/review-queue"),
        method="GET",
    )
    return _parse_human_review_queue_response(
        _send_api_request(http_request, timeout_seconds=timeout_seconds)
    )


def _build_api_url(api_base_url: str, path: str) -> str:
    """规范化页面填写的服务地址，避免重复或遗漏路径分隔符。"""

    normalized_base_url = api_base_url.strip().rstrip("/")
    if not normalized_base_url:
        raise TriageApiConnectionError("请输入 FastAPI 服务地址。")
    return f"{normalized_base_url}{path}"


def _send_api_request(http_request: Request, *, timeout_seconds: float) -> bytes:
    """发送本地 HTTP 请求；HTTP 错误响应仍交由后续 Pydantic 模型判断。"""

    try:
        # 使用 Python 标准库即可访问本地 API，避免为页面额外引入 HTTP 客户端依赖。
        with urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except HTTPError as error:
        # 422 等 HTTP 错误也可能携带合规的人工复核 JSON，继续进行 Pydantic 校验。
        response_body = error.read()
    except (URLError, OSError, ValueError) as error:
        raise TriageApiConnectionError(
            "无法连接 FastAPI 服务。请确认它仍在运行，并检查服务地址。"
        ) from error


    return response_body


def _parse_json_payload(response_body: bytes) -> Any:
    """把 HTTP 响应解码为 JSON 值，拒绝非 JSON 或非 UTF-8 内容。"""

    try:
        return json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TriageApiResponseError(
            "API 返回的内容不是有效 JSON，页面不会展示未经校验的结果。"
        ) from error


def _parse_service_case_result(response_body: bytes) -> ServiceCaseResult:
    """解析 HTTP 响应，并拒绝任何不符合核心输出契约的内容。"""

    try:
        return ServiceCaseResult.model_validate(_parse_json_payload(response_body))
    except ValidationError as error:
        raise TriageApiResponseError(
            "API 返回的 JSON 未通过 Pydantic 安全校验，页面不会展示该结果。"
        ) from error


def _parse_case_history_response(response_body: bytes) -> CaseHistoryResponse:
    """解析案件历史响应，只允许页面使用经过 Pydantic 校验的记录。"""

    try:
        return CaseHistoryResponse.model_validate(_parse_json_payload(response_body))
    except ValidationError as error:
        raise TriageApiResponseError(
            "案件历史 JSON 未通过 Pydantic 安全校验，页面不会展示该结果。"
        ) from error


def _parse_human_review_queue_response(
    response_body: bytes,
) -> HumanReviewQueueResponse:
    """解析待人工复核列表响应，只允许展示安全标记一致的案件。"""

    try:
        return HumanReviewQueueResponse.model_validate(
            _parse_json_payload(response_body)
        )
    except ValidationError as error:
        raise TriageApiResponseError(
            "待人工复核列表 JSON 未通过 Pydantic 安全校验，页面不会展示该结果。"
        ) from error
