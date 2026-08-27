"""通过 OpenAI-compatible Chat Completions 接口调用视觉模型。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.llm.openai_compatible_provider import (
    HttpOpener,
    OpenAICompatibleProviderSettings,
)
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProvider,
    VisionProviderError,
    VisionResponse,
)


class OpenAICompatibleVisionProvider(VisionProvider):
    """封装兼容 Chat Completions 多模态消息格式的视觉模型。"""

    def __init__(
        self,
        settings: OpenAICompatibleProviderSettings,
        *,
        http_opener: HttpOpener | None = None,
    ) -> None:
        self._settings = settings
        # 测试可以注入内存 HTTP 响应，默认才使用标准库联网。
        self._http_opener = http_opener or urlopen

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "OpenAICompatibleVisionProvider":
        """读取与文本模型相同的 OpenAI-compatible 环境变量。"""

        return cls(OpenAICompatibleProviderSettings.from_environment(environ))

    def generate_json(self, request: VisionExtractionRequest) -> VisionResponse:
        """发送图片和 JSON Schema，并只返回未解析的模型文本。"""

        # data URL 只在发送请求时临时构造，不能写入诊断或历史记录。
        image_url = (
            f"data:{request.image.media_type};base64,{request.image.data_base64}"
        )
        payload = self._build_payload(request, image_url=image_url)
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
            # 不回显第三方响应体，避免密钥或服务端内容进入本地日志。
            raise VisionProviderError(
                f"OpenAI-compatible 视觉服务返回 HTTP {error.code}。"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise VisionProviderError(
                "无法连接 OpenAI-compatible 视觉模型服务。"
            ) from error

        return VisionResponse(
            content=self._extract_response_content(response_body),
            provider_name="openai_compatible_vision",
            model_name=self._settings.model,
        )

    def _build_payload(
        self,
        request: VisionExtractionRequest,
        *,
        image_url: str | None = None,
    ) -> dict[str, object]:
        """构造标准 image_url 消息，并强制模型返回 JSON 对象。"""

        schema_instruction = (
            "你必须只返回一个有效 JSON 对象，并严格符合以下 JSON Schema：\n"
            + json.dumps(request.json_schema, ensure_ascii=False)
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": schema_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                            or (
                                f"data:{request.image.media_type};base64,"
                                f"{request.image.data_base64}"
                            ),
                        },
                    },
                ],
            },
        ]
        if self._settings.structured_output_mode == "json_schema":
            response_format: dict[str, object] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}

        return {
            "model": self._settings.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": response_format,
        }

    @staticmethod
    def _extract_response_content(response_body: object) -> str:
        """只接受 choices[0].message.content 字符串。"""

        if not isinstance(response_body, (bytes, bytearray)):
            raise VisionProviderError("视觉服务返回了非字节形式的 HTTP 响应。")
        try:
            payload = json.loads(bytes(response_body).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VisionProviderError("视觉服务返回的内容不是有效 JSON。") from error
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VisionProviderError(
                "视觉服务响应缺少 choices[0].message.content。"
            ) from error
        if not isinstance(content, str):
            raise VisionProviderError(
                "视觉服务响应中的 message.content 不是字符串。"
            )
        return content
