"""验证 OpenAI-compatible 视觉提供方的请求封装和安全异常处理。"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from urllib.error import HTTPError

import pytest

from app.llm.openai_compatible_provider import OpenAICompatibleProviderSettings
from app.schemas.models import ImageAttachment, VisionObservation
from app.vision.openai_compatible_provider import OpenAICompatibleVisionProvider
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProviderError,
)


class FakeHttpResponse:
    """为单元测试模拟可用作上下文管理器的 HTTP 响应。"""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        """返回预设响应字节。"""

        return self._body


def build_request() -> VisionExtractionRequest:
    """构造最小图片请求，不依赖真实照片或外部服务。"""

    demo_bytes = b"vision-provider-demo"
    return VisionExtractionRequest(
        image=ImageAttachment(
            media_type="image/png",
            data_base64=base64.b64encode(demo_bytes).decode("ascii"),
            filename="demo.png",
        ),
        prompt="请只返回结构化图片观察。",
        schema_name="vision_observation",
        json_schema=VisionObservation.model_json_schema(),
    )


def build_settings(
    *,
    structured_output_mode: str = "json_object",
) -> OpenAICompatibleProviderSettings:
    """生成不含真实密钥的固定测试配置。"""

    return OpenAICompatibleProviderSettings(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-vision-model",
        structured_output_mode=structured_output_mode,
    )


def test_provider_posts_multimodal_chat_completions_request() -> None:
    """提供方应发送图片 data URL、JSON 约束和授权请求。"""

    captured: dict[str, Any] = {}

    def fake_opener(http_request: Any, timeout: float) -> FakeHttpResponse:
        captured["url"] = http_request.full_url
        captured["body"] = http_request.data
        captured["headers"] = dict(http_request.header_items())
        captured["timeout"] = timeout
        return FakeHttpResponse(
            json.dumps(
                {
                    "choices": [
                        {"message": {"content": '{"description":"demo"}'}},
                    ]
                }
            ).encode("utf-8")
        )

    provider = OpenAICompatibleVisionProvider(
        build_settings(),
        http_opener=fake_opener,
    )
    response = provider.generate_json(build_request())

    payload = json.loads(captured["body"].decode("utf-8"))
    normalized_headers = {
        name.lower(): value for name, value in captured["headers"].items()
    }
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 30.0
    assert normalized_headers["authorization"] == "Bearer test-key"
    assert payload["model"] == "test-vision-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"][0] == {
        "type": "text",
        "text": "请只返回结构化图片观察。",
    }
    assert image_url.startswith("data:image/png;base64,")
    assert image_url.endswith(
        base64.b64encode(b"vision-provider-demo").decode("ascii")
    )
    assert response.content == '{"description":"demo"}'
    assert response.provider_name == "openai_compatible_vision"
    assert response.model_name == "test-vision-model"


def test_provider_can_use_json_schema_response_format() -> None:
    """服务明确支持 JSON Schema 时，应发送完整的视觉输出约束。"""

    captured: dict[str, Any] = {}

    def fake_opener(http_request: Any, timeout: float) -> FakeHttpResponse:
        captured["body"] = http_request.data
        return FakeHttpResponse(
            b'{"choices":[{"message":{"content":"{}"}}]}'
        )

    provider = OpenAICompatibleVisionProvider(
        build_settings(structured_output_mode="json_schema"),
        http_opener=fake_opener,
    )
    provider.generate_json(build_request())

    payload = json.loads(captured["body"].decode("utf-8"))
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == (
        "vision_observation"
    )
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_provider_rejects_invalid_response_shape() -> None:
    """缺少标准内容字段时不得伪造视觉输出，必须抛出统一异常。"""

    provider = OpenAICompatibleVisionProvider(
        build_settings(),
        http_opener=lambda *_args, **_kwargs: FakeHttpResponse(
            b'{"choices":[]}'
        ),
    )

    with pytest.raises(VisionProviderError, match="choices"):
        provider.generate_json(build_request())


def test_provider_hides_http_error_response_body() -> None:
    """第三方 HTTP 错误正文不应进入异常或人工复核记录。"""

    def fake_opener(*_args: object, **_kwargs: object) -> FakeHttpResponse:
        raise HTTPError(
            url="https://example.test/v1/chat/completions",
            code=429,
            msg="too many requests",
            hdrs=None,
            fp=BytesIO(b"do-not-record-this-provider-response"),
        )

    provider = OpenAICompatibleVisionProvider(
        build_settings(),
        http_opener=fake_opener,
    )

    with pytest.raises(VisionProviderError, match="HTTP 429") as error_info:
        provider.generate_json(build_request())

    assert "do-not-record" not in str(error_info.value)
