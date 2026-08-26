"""模型提供方接口的数据校验测试。"""

import pytest
from pydantic import ValidationError

from app.llm.provider import (
    ChatMessage,
    LLMProvider,
    StructuredGenerationRequest,
)


def test_structured_request_accepts_json_schema() -> None:
    """有效请求必须保留消息、Schema 名称和 JSON Schema。"""

    request = StructuredGenerationRequest(
        messages=[ChatMessage(role="user", content="卫生间没水")],
        schema_name="service_case_result",
        json_schema={"type": "object"},
    )

    assert request.temperature == 0.0
    assert request.messages[0].content == "卫生间没水"
    assert request.json_schema == {"type": "object"}


def test_structured_request_rejects_empty_json_schema() -> None:
    """空 Schema 会导致模型生成自由文本，因此必须被拒绝。"""

    with pytest.raises(ValidationError, match="json_schema"):
        StructuredGenerationRequest(
            messages=[ChatMessage(role="user", content="卫生间没水")],
            schema_name="service_case_result",
            json_schema={},
        )


def test_provider_interface_cannot_be_created_directly() -> None:
    """抽象接口本身不能调用，必须由模拟或真实提供方实现。"""

    with pytest.raises(TypeError):
        LLMProvider()
