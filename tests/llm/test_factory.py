"""验证环境变量模型工厂默认无网络调用，并拒绝不完整配置。"""

import pytest

from app.llm.demo_provider import DemoLLMProvider
from app.llm.factory import build_model_provider_from_environment
from app.llm.openai_compatible_provider import OpenAICompatibleLLMProvider
from app.llm.provider import ModelProviderError


def test_factory_uses_free_demo_provider_by_default() -> None:
    """未配置 LLM_PROVIDER 时必须保持免费本地演示模式。"""

    provider = build_model_provider_from_environment({})

    assert isinstance(provider, DemoLLMProvider)


def test_factory_builds_openai_compatible_provider_from_complete_settings() -> None:
    """只有完整显式配置时才创建外部模型提供方，创建本身不发网络请求。"""

    provider = build_model_provider_from_environment(
        {
            "LLM_PROVIDER": "openai_compatible",
            "OPENAI_COMPATIBLE_BASE_URL": "https://example.test/v1",
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENAI_COMPATIBLE_MODEL": "test-model",
        }
    )

    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_factory_rejects_missing_external_model_settings() -> None:
    """外部模型配置不完整时应失败，避免启动后才产生难以解释的网络错误。"""

    with pytest.raises(ModelProviderError, match="OPENAI_COMPATIBLE_API_KEY"):
        build_model_provider_from_environment(
            {
                "LLM_PROVIDER": "openai_compatible",
                "OPENAI_COMPATIBLE_BASE_URL": "https://example.test/v1",
                "OPENAI_COMPATIBLE_MODEL": "test-model",
            }
        )


def test_factory_rejects_unknown_provider_name() -> None:
    """拼写错误的提供方名称必须明确报错，不能悄悄改为其他模型。"""

    with pytest.raises(ModelProviderError, match="demo"):
        build_model_provider_from_environment({"LLM_PROVIDER": "unknown"})
