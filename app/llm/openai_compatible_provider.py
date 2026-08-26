"""通过 OpenAI-compatible Chat Completions API 调用可替换的模型。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)

from app.llm.provider import (
    LLMProvider,
    LLMResponse,
    ModelProviderError,
    ProviderSchema,
    StructuredGenerationRequest,
)


# 传入测试替身时只需要实现与 urllib.request.urlopen 相同的调用形式。
HttpOpener = Callable[..., Any]


class OpenAICompatibleProviderSettings(ProviderSchema):
    """OpenAI-compatible 模型服务的最小连接配置。"""

    # base_url 约定为 API 根地址，例如 https://example.com/v1，而非完整接口地址。
    base_url: str = Field(min_length=1, max_length=500)
    # SecretStr 能避免配置对象被打印时意外显示 API 密钥。
    api_key: SecretStr
    model: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    structured_output_mode: Literal["json_object", "json_schema"] = "json_object"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """只接受明确的 HTTP(S) API 根地址，避免拼接出不可预期的请求地址。"""

        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return normalized

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        """即使配置由代码直接创建，也不允许空白 API 密钥。"""

        if not value.get_secret_value().strip():
            raise ValueError("api_key must not be blank")
        return value

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "OpenAICompatibleProviderSettings":
        """从环境变量读取配置；缺少密钥时拒绝启动，不回退为伪造结果。"""

        environment = os.environ if environ is None else environ
        required_names = {
            "OPENAI_COMPATIBLE_BASE_URL": "base_url",
            "OPENAI_COMPATIBLE_API_KEY": "api_key",
            "OPENAI_COMPATIBLE_MODEL": "model",
        }
        missing_names = [
            name
            for name in required_names
            if not environment.get(name, "").strip()
        ]
        if missing_names:
            raise ModelProviderError(
                "缺少 OpenAI-compatible 模型配置："
                + ", ".join(missing_names)
            )

        try:
            return cls(
                base_url=environment["OPENAI_COMPATIBLE_BASE_URL"],
                api_key=environment["OPENAI_COMPATIBLE_API_KEY"],
                model=environment["OPENAI_COMPATIBLE_MODEL"],
                timeout_seconds=environment.get(
                    "OPENAI_COMPATIBLE_TIMEOUT_SECONDS",
                    "30",
                ),
                structured_output_mode=environment.get(
                    "OPENAI_COMPATIBLE_STRUCTURED_OUTPUT_MODE",
                    "json_object",
                ),
            )
        except ValidationError as error:
            # 只保留字段名和验证信息，不能把可能包含密钥的输入拼进异常文本。
            messages = [
                f"{'.'.join(str(part) for part in detail['loc'])}: "
                f"{detail['msg']}"
                for detail in error.errors(include_url=False)
            ]
            raise ModelProviderError(
                "OpenAI-compatible 模型配置无效：" + "; ".join(messages)
            ) from error


class OpenAICompatibleLLMProvider(LLMProvider):
    """封装标准 Chat Completions 接口，统一返回未解析的模型 JSON 文本。"""

    def __init__(
        self,
        settings: OpenAICompatibleProviderSettings,
        *,
        http_opener: HttpOpener | None = None,
    ) -> None:
        self._settings = settings
        # 测试可注入内存响应，正常运行时才使用标准库的网络实现。
        self._http_opener = http_opener or urlopen

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "OpenAICompatibleLLMProvider":
        """从环境变量创建提供方，供 API 启动工厂调用。"""

        return cls(OpenAICompatibleProviderSettings.from_environment(environ))

    def generate_json(
        self,
        request: StructuredGenerationRequest,
    ) -> LLMResponse:
        """请求模型生成 JSON；网络或响应异常统一转换为安全的提供方错误。"""

        request_body = json.dumps(
            self._build_payload(request),
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = Request(
            url=f"{self._settings.base_url}/chat/completions",
            data=request_body,
            headers={
                "Accept": "application/json",
                "Authorization": (
                    "Bearer " + self._settings.api_key.get_secret_value()
                ),
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        try:
            with self._http_opener(
                http_request,
                timeout=self._settings.timeout_seconds,
            ) as http_response:
                response_body = http_response.read()
        except HTTPError as error:
            # 不回显响应体，避免第三方服务返回的内容污染本地审计记录。
            raise ModelProviderError(
                "OpenAI-compatible 模型服务返回 HTTP "
                f"{error.code}。"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise ModelProviderError(
                "无法连接 OpenAI-compatible 模型服务。"
            ) from error

        return LLMResponse(
            content=self._extract_response_content(response_body),
            provider_name="openai_compatible",
            # 使用配置中的模型名，避免把服务端返回的未校验元数据写入审计结果。
            model_name=self._settings.model,
        )

    def _build_payload(
        self,
        request: StructuredGenerationRequest,
    ) -> dict[str, object]:
        """按兼容接口约定构造请求体，并始终要求 JSON 对象输出。"""

        messages = [message.model_dump(mode="json") for message in request.messages]
        if self._settings.structured_output_mode == "json_object":
            # json_object 兼容面较广，因此额外把 Schema 放入系统消息中供模型遵守。
            schema_instruction = (
                "你必须只返回一个有效 JSON 对象，并严格符合以下 JSON Schema：\n"
                + json.dumps(request.json_schema, ensure_ascii=False)
            )
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": schema_instruction,
                },
            )
            response_format: dict[str, object] = {"type": "json_object"}
        else:
            # 仅在服务明确支持 OpenAI JSON Schema structured output 时启用此模式。
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                },
            }

        return {
            "model": self._settings.model,
            "messages": messages,
            "temperature": request.temperature,
            "response_format": response_format,
        }

    @staticmethod
    def _extract_response_content(response_body: object) -> str:
        """只接受标准响应中的 choices[0].message.content 字符串。"""

        if not isinstance(response_body, (bytes, bytearray)):
            raise ModelProviderError("模型服务返回了非字节形式的 HTTP 响应。")

        try:
            payload = json.loads(bytes(response_body).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                "模型服务返回的内容不是有效 JSON。"
            ) from error

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelProviderError(
                "模型服务响应缺少 choices[0].message.content。"
            ) from error

        if not isinstance(content, str):
            raise ModelProviderError(
                "模型服务响应中的 choices[0].message.content 不是字符串。"
            )
        return content
