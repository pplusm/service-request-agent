"""根据环境变量选择默认演示模型或可替换的外部模型。"""

from __future__ import annotations

import os
from collections.abc import Mapping

from app.llm.demo_provider import DemoLLMProvider
from app.llm.openai_compatible_provider import OpenAICompatibleLLMProvider
from app.llm.provider import LLMProvider, ModelProviderError


def build_model_provider_from_environment(
    environ: Mapping[str, str] | None = None,
) -> LLMProvider:
    """创建模型提供方；未设置时始终使用免费、无网络调用的演示模型。"""

    environment = os.environ if environ is None else environ
    provider_name = environment.get("LLM_PROVIDER", "demo").strip().lower()

    if provider_name in {"", "demo"}:
        return DemoLLMProvider()
    if provider_name == "openai_compatible":
        return OpenAICompatibleLLMProvider.from_environment(environment)

    raise ModelProviderError(
        "LLM_PROVIDER 只支持 demo 或 openai_compatible。"
    )
