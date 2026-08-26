"""验证 OpenAI-compatible 提供方的请求封装和安全异常处理。"""

import json
from io import BytesIO
from typing import Any
from urllib.error import HTTPError

import pytest

from app.llm.openai_compatible_provider import (
    OpenAICompatibleLLMProvider,
    OpenAICompatibleProviderSettings,
)
from app.llm.provider import (
    ChatMessage,
    ModelProviderError,
    StructuredGenerationRequest,
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


def build_request() -> StructuredGenerationRequest:
    """构造最小结构化生成请求，避免测试依赖实际模型或网络。"""

    return StructuredGenerationRequest(
        messages=[ChatMessage(role="user", content="卫生间没水")],
        schema_name="service_case_result",
        json_schema={
            "type": "object",
            "properties": {"risk": {"type": "string"}},
        },
    )


def build_settings(
    *,
    structured_output_mode: str = "json_object",
) -> OpenAICompatibleProviderSettings:
    """生成不含真实密钥的固定测试配置。"""

    return OpenAICompatibleProviderSettings(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        structured_output_mode=structured_output_mode,
    )


def test_provider_posts_standard_chat_completions_request() -> None:
    """提供方应发送授权请求、JSON 对象约束和原始模型消息。"""

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
                        {"message": {"content": '{"risk":"low"}'}}
                    ]
                }
            ).encode("utf-8")
        )

    provider = OpenAICompatibleLLMProvider(
        build_settings(),
        http_opener=fake_opener,
    )
    response = provider.generate_json(build_request())

    payload = json.loads(captured["body"].decode("utf-8"))
    normalized_headers = {
        name.lower(): value for name, value in captured["headers"].items()
    }
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 30.0
    assert normalized_headers["authorization"] == "Bearer test-key"
    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"
    assert '"risk"' in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "卫生间没水"}
    assert response.content == '{"risk":"low"}'
    assert response.provider_name == "openai_compatible"
    assert response.model_name == "test-model"


def test_provider_can_use_json_schema_response_format() -> None:
    """服务明确支持 JSON Schema 时，应按兼容协议发送完整约束。"""

    captured: dict[str, Any] = {}

    def fake_opener(http_request: Any, timeout: float) -> FakeHttpResponse:
        captured["body"] = http_request.data
        return FakeHttpResponse(
            b'{"choices":[{"message":{"content":"{}"}}]}'
        )

    provider = OpenAICompatibleLLMProvider(
        build_settings(structured_output_mode="json_schema"),
        http_opener=fake_opener,
    )
    provider.generate_json(build_request())

    payload = json.loads(captured["body"].decode("utf-8"))
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "service_case_result",
            "schema": {
                "type": "object",
                "properties": {"risk": {"type": "string"}},
            },
            "strict": True,
        },
    }


def test_provider_rejects_invalid_response_shape() -> None:
    """缺少标准内容字段时不得伪造模型输出，必须抛出统一异常。"""

    provider = OpenAICompatibleLLMProvider(
        build_settings(),
        http_opener=lambda *_args, **_kwargs: FakeHttpResponse(b'{"choices":[]}'),
    )

    with pytest.raises(ModelProviderError, match="choices"):
        provider.generate_json(build_request())


def test_provider_hides_http_error_response_body() -> None:
    """第三方 HTTP 错误内容不应写入异常，避免把未知内容保留到审计字段。"""

    def fake_opener(*_args: object, **_kwargs: object) -> FakeHttpResponse:
        raise HTTPError(
            url="https://example.test/v1/chat/completions",
            code=429,
            msg="too many requests",
            hdrs=None,
            fp=BytesIO(b"do-not-record-this-provider-response"),
        )

    provider = OpenAICompatibleLLMProvider(
        build_settings(),
        http_opener=fake_opener,
    )

    with pytest.raises(ModelProviderError, match="HTTP 429") as error_info:
        provider.generate_json(build_request())

    assert "do-not-record" not in str(error_info.value)


def test_settings_reject_blank_api_key_when_created_directly() -> None:
    """直接创建配置也必须拒绝空密钥，不能绕过环境变量入口的安全检查。"""

    with pytest.raises(ValueError, match="api_key"):
        OpenAICompatibleProviderSettings(
            base_url="https://example.test/v1",
            api_key="   ",
            model="test-model",
        )
