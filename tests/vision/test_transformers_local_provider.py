"""验证本地 Transformers 视觉提供方的延迟加载与安全边界。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.models import ImageAttachment, VisionObservation
from app.vision.demo_provider import DemoVisionProvider
from app.vision.factory import build_vision_provider_from_environment
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProviderError,
)
from app.vision.transformers_local_provider import (
    LocalQwenVisionSettings,
    TransformersLocalVisionProvider,
)


class FakeLocalQwenBackend:
    """替代真实 7.5 GB 模型，记录调用参数供单元测试断言。"""

    def __init__(
        self,
        content: str,
        *,
        load_error: Exception | None = None,
    ) -> None:
        self.content = content
        self.load_error = load_error
        self.load_calls = 0
        self.generate_calls = 0
        self.images: list[Any] = []
        self.requests: list[VisionExtractionRequest] = []

    def load(self, _settings: LocalQwenVisionSettings) -> tuple[Any, Any]:
        """返回普通对象，证明测试不会真正加载 Transformers 模型。"""

        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return object(), object()

    def generate(
        self,
        *,
        model: Any,
        processor: Any,
        image: Any,
        request: VisionExtractionRequest,
        settings: LocalQwenVisionSettings,
    ) -> str:
        """记录输入并返回预设模型文本。"""

        assert model is not None
        assert processor is not None
        assert settings.max_new_tokens == 512
        self.generate_calls += 1
        self.images.append(image)
        self.requests.append(request)
        return self.content


def build_request() -> VisionExtractionRequest:
    """构造仅含演示字节的视觉请求，不依赖真实照片。"""

    return VisionExtractionRequest(
        image=ImageAttachment(
            media_type="image/png",
            data_base64=base64.b64encode(b"test-image-bytes").decode("ascii"),
            filename="test.png",
        ),
        prompt="请只返回结构化图片观察。",
        schema_name="vision_observation",
        json_schema=VisionObservation.model_json_schema(),
    )


def build_settings(tmp_path: Path) -> LocalQwenVisionSettings:
    """使用 pytest 临时目录模拟已经存在的本地模型目录。"""

    return LocalQwenVisionSettings(model_path=tmp_path)


def build_valid_observation_json() -> str:
    """返回能通过最终 Pydantic 校验的最小真实视觉观察 JSON。"""

    return json.dumps(
        VisionObservation(
            description="图片中可见一盏损坏的路灯。",
            objects=["路灯"],
            hazard_signals=["照明设备损坏"],
            uncertainty_notes=["无法确认具体地点。"],
            confidence=0.8,
            is_demo_observation=False,
        ).model_dump(mode="json"),
        ensure_ascii=False,
    )


def test_factory_keeps_demo_provider_by_default() -> None:
    """未设置环境变量时必须维持原来的免费演示视觉模型。"""

    provider = build_vision_provider_from_environment({})

    assert isinstance(provider, DemoVisionProvider)


def test_factory_builds_local_provider_without_loading_model(tmp_path: Path) -> None:
    """工厂创建本地提供方时不能立刻加载数 GB 权重。"""

    provider = build_vision_provider_from_environment(
        {
            "VISION_PROVIDER": "transformers_local",
            "LOCAL_QWEN_VISION_MODEL_PATH": str(tmp_path),
        }
    )

    assert isinstance(provider, TransformersLocalVisionProvider)


def test_factory_rejects_missing_local_model_path() -> None:
    """本地模型路径遗漏时应报清晰配置错误，不能伪造 demo 结果。"""

    with pytest.raises(
        VisionProviderError,
        match="LOCAL_QWEN_VISION_MODEL_PATH",
    ):
        build_vision_provider_from_environment(
            {"VISION_PROVIDER": "transformers_local"}
        )


def test_local_provider_loads_lazily_and_unwraps_single_json_code_fence(
    tmp_path: Path,
) -> None:
    """模型只在首张图片加载一次，完整 JSON 代码块可交给 Pydantic 校验。"""

    raw_json = build_valid_observation_json()
    backend = FakeLocalQwenBackend(f"```json\n{raw_json}\n```")
    decoded_image = object()
    provider = TransformersLocalVisionProvider(
        build_settings(tmp_path),
        backend=backend,
        image_decoder=lambda _image: decoded_image,
    )

    assert backend.load_calls == 0
    first_response = provider.generate_json(build_request())
    second_response = provider.generate_json(build_request())

    assert backend.load_calls == 1
    assert backend.generate_calls == 2
    assert backend.images == [decoded_image, decoded_image]
    assert first_response.content == raw_json
    assert second_response.content == raw_json
    assert first_response.provider_name == "transformers_local_vision"
    assert first_response.model_name == tmp_path.name
    # Provider 不自行构造观察对象；最终仍由与 Agent 相同的 Pydantic 模型校验。
    assert (
        VisionObservation.model_validate_json(first_response.content).confidence
        == 0.8
    )


def test_local_provider_model_loading_failure_hides_internal_error(
    tmp_path: Path,
) -> None:
    """模型加载异常必须转成统一错误，不能把内部细节写入诊断。"""

    backend = FakeLocalQwenBackend(
        build_valid_observation_json(),
        load_error=RuntimeError("internal model path and image data must stay hidden"),
    )
    provider = TransformersLocalVisionProvider(
        build_settings(tmp_path),
        backend=backend,
        image_decoder=lambda _image: object(),
    )

    with pytest.raises(VisionProviderError, match="加载失败") as error_info:
        provider.generate_json(build_request())

    assert "internal model path" not in str(error_info.value)
    assert backend.generate_calls == 0


def test_local_provider_does_not_repair_invalid_json_content(tmp_path: Path) -> None:
    """非 JSON 模型输出不做字段修补，后续 Pydantic 校验会使案件进入人工复核。"""

    backend = FakeLocalQwenBackend("```json\nthis is not JSON\n```")
    provider = TransformersLocalVisionProvider(
        build_settings(tmp_path),
        backend=backend,
        image_decoder=lambda _image: object(),
    )

    response = provider.generate_json(build_request())

    assert response.content == "this is not JSON"
    with pytest.raises(ValidationError):
        VisionObservation.model_validate_json(response.content)
