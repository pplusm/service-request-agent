"""定义可替换的模型提供方统一接口。"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 模型模块单独使用严格校验，避免模型调用参数中混入未定义字段。
class ProviderSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChatMessage(ProviderSchema):
    """发送给聊天模型的一条消息。"""

    # 第一阶段只需要这三种 OpenAI-compatible 通用消息角色。
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class StructuredGenerationRequest(ProviderSchema):
    """要求模型返回符合指定 JSON Schema 的生成请求。"""

    # 一次调用至少要有一条消息；上限防止意外构造过大的请求。
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)

    # 后续会传入 Pydantic 的 model_json_schema() 结果，供模型服务约束输出。
    schema_name: str = Field(min_length=1, max_length=100)
    json_schema: dict[str, Any]

    # 低温度使演示分类结果更稳定；不同模型提供方都能识别这一通用参数。
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_json_schema(self) -> "StructuredGenerationRequest":
        """拒绝空 JSON Schema，确保调用方不会请求自由文本输出。"""

        if not self.json_schema:
            raise ValueError("json_schema must not be empty")
        return self


class LLMResponse(ProviderSchema):
    """模型提供方返回的原始文本及其来源信息。"""

    # content 保留原始字符串；后续 Agent 必须再用 Pydantic 解析和校验它。
    content: str = Field(max_length=50_000)
    provider_name: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=200)


class ModelProviderError(RuntimeError):
    """模型提供方调用失败时使用的统一异常类型。"""


class LLMProvider(ABC):
    """所有模型提供方都必须实现的统一接口。"""

    @abstractmethod
    def generate_json(
        self, request: StructuredGenerationRequest
    ) -> LLMResponse:
        """返回模型原始输出，不能在提供方层跳过 Pydantic 校验。"""

        raise NotImplementedError
