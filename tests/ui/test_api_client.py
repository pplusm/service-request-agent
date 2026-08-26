"""验证 Streamlit 页面调用 API 时不会展示未通过 Pydantic 校验的结果。"""

import json
from typing import Any
from urllib.error import URLError

import pytest

from app.schemas.models import ServiceCaseResult
from app.ui.api_client import (
    TriageApiConnectionError,
    TriageApiResponseError,
    submit_triage_request,
)


class FakeHttpResponse:
    """模拟 urllib 响应对象，让测试不依赖真实启动的 Web 服务。"""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def build_valid_case_result() -> ServiceCaseResult:
    """构造一份可通过核心 Pydantic 模型校验的 API 结果。"""

    return ServiceCaseResult.model_validate(
        {
            "request_id": "ui_demo_001",
            "scenario": "scenic_service",
            "entities": {
                "location": "西门",
                "facility_name": "照明设施",
                "visitor_condition": None,
                "estimated_affected_count": None,
                "event_time_description": None,
                "missing_fields": [],
            },
            "classification": {
                "event_type": "facility_fault",
                "confidence": 0.9,
                "evidence": ["西门照明故障"],
            },
            "risk": {
                "level": "low",
                "risk_factors": [],
                "summary": "演示性低风险设施故障。",
            },
            "knowledge_references": [
                {
                    "source_id": "demo_facility_001",
                    "source_title": "Demo facility guidance",
                    "source_path": "data/scenic_service/knowledge/demo_facility.md",
                    "excerpt": "演示资料：照明故障属于设施故障示例。",
                    "relevance_score": 0.9,
                    "is_demo_source": True,
                }
            ],
            "action_plan": [
                {
                    "step": 1,
                    "suggested_action": "创建演示性照明设施维护跟进建议。",
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
    )


def test_submit_triage_request_sends_the_expected_post_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """页面应把表单内容发送到固定分诊接口，并解析合规 JSON。"""

    captured: dict[str, Any] = {}
    response_body = json.dumps(
        build_valid_case_result().model_dump(mode="json"),
        ensure_ascii=False,
    ).encode("utf-8")

    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHttpResponse(response_body)

    monkeypatch.setattr("app.ui.api_client.urlopen", fake_urlopen)

    result = submit_triage_request(
        api_base_url="http://127.0.0.1:8000/",
        request_id="ui_demo_001",
        text="西门照明故障",
    )

    assert result.request_id == "ui_demo_001"
    assert captured["url"] == "http://127.0.0.1:8000/api/v1/triage"
    assert captured["payload"] == {
        "request_id": "ui_demo_001",
        "text": "西门照明故障",
    }
    assert captured["timeout"] == 15.0


def test_submit_triage_request_rejects_an_invalid_api_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """响应缺少核心字段时，页面必须拒绝展示它。"""

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeHttpResponse:
        return FakeHttpResponse(b'{"request_id": "invalid"}')

    monkeypatch.setattr("app.ui.api_client.urlopen", fake_urlopen)

    with pytest.raises(TriageApiResponseError):
        submit_triage_request(
            api_base_url="http://127.0.0.1:8000",
            request_id="ui_demo_002",
            text="西门照明故障",
        )


def test_submit_triage_request_explains_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地 API 未启动时，页面应给出可理解的连接错误。"""

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> FakeHttpResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("app.ui.api_client.urlopen", fake_urlopen)

    with pytest.raises(TriageApiConnectionError, match="无法连接 FastAPI 服务"):
        submit_triage_request(
            api_base_url="http://127.0.0.1:8000",
            request_id="ui_demo_003",
            text="西门照明故障",
        )
