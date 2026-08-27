"""定义视觉模型提供方的统一、可替换接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field, model_validator

from app.llm.provider import ProviderSchema
from app.schemas.models import ImageAttachment


class VisionExtractionRequest(ProviderSchema):
    """传给视觉模型的图片和结构化输出约束。"""

    image: ImageAttachment
    prompt: str = Field(min_length=1, max_length=12_000)
    schema_name: str = Field(min_length=1, max_length=100)
    json_schema: dict[str, Any]

    @model_validator(mode="after")
    def validate_json_schema(self) -> "VisionExtractionRequest":
        """拒绝空 Schema，确保视觉提供方不会退化成自由描述。"""

        if not self.json_schema:
            raise ValueError("json_schema must not be empty")
        return self


class VisionResponse(ProviderSchema):
    """视觉提供方返回的原始 JSON 文本及服务标识。"""

    content: str = Field(max_length=50_000)
    provider_name: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=200)


class VisionProviderError(RuntimeError):
    """视觉模型调用失败时使用的统一异常类型。"""


class VisionProvider(ABC):
    """所有视觉模型实现都必须遵守的统一接口。"""

    @abstractmethod
    def generate_json(self, request: VisionExtractionRequest) -> VisionResponse:
        """返回尚未解析的 JSON 文本，解析必须由 Agent 的 Pydantic 层完成。"""

        raise NotImplementedError

