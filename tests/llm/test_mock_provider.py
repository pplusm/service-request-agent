"""验证模拟模型提供方的测试。"""

import pytest

from app.llm.mock_provider import MockLLMProvider
from app.llm.provider import (
    ChatMessage,
    ModelProviderError,
    StructuredGenerationRequest,
)


def build_request() -> StructuredGenerationRequest:
    """构造一条供模拟模型使用的最小结构化生成请求。"""

    return StructuredGenerationRequest(
        messages=[ChatMessage(role="user", content="卫生间没水")],
        schema_name="service_case_result",
        json_schema={"type": "object"},
    )


def test_mock_provider_returns_configured_response_and_records_request() -> None:
    """模拟模型应返回指定文本，并保存本次调用的独立快照。"""

    provider = MockLLMProvider(
        response_content='{"event_type": "facility_fault"}',
        model_name="test-model",
    )
    request = build_request()

    response = provider.generate_json(request)
    request.messages[0].content = "后续修改不应影响记录"

    assert response.content == '{"event_type": "facility_fault"}'
    assert response.provider_name == "mock"
    assert response.model_name == "test-model"
    assert provider.requests[0].messages[0].content == "卫生间没水"


def test_mock_provider_can_return_invalid_json_for_fallback_tests() -> None:
    """模拟模型允许返回非法 JSON，供后续人工复核流程测试。"""

    provider = MockLLMProvider(response_content="这不是 JSON")

    response = provider.generate_json(build_request())

    assert response.content == "这不是 JSON"


def test_mock_provider_can_simulate_provider_error() -> None:
    """模型服务异常时，调用方应收到统一异常类型。"""

    provider = MockLLMProvider(error_message="演示连接失败")

    with pytest.raises(ModelProviderError, match="演示连接失败"):
        provider.generate_json(build_request())
