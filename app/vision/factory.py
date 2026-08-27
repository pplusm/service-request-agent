"""根据环境变量创建视觉模型提供方。"""

from __future__ import annotations

import os
from collections.abc import Mapping

from app.vision.demo_provider import DemoVisionProvider
from app.vision.openai_compatible_provider import OpenAICompatibleVisionProvider
from app.vision.provider import VisionProvider, VisionProviderError


def build_vision_provider_from_environment(
    environ: Mapping[str, str] | None = None,
) -> VisionProvider:
    """默认返回本地 demo；真实视觉模型可通过同一接口后续替换。"""

    environment = os.environ if environ is None else environ
    provider_name = environment.get("VISION_PROVIDER", "demo").strip().lower()
    if provider_name in {"", "demo"}:
        return DemoVisionProvider()
    if provider_name == "openai_compatible":
        return OpenAICompatibleVisionProvider.from_environment(environment)

    raise VisionProviderError(
        "VISION_PROVIDER 只支持 demo 或 openai_compatible。"
    )
