"""提供仅用于本地测试的模拟模型实现。"""

from app.llm.provider import (
    LLMProvider,
    LLMResponse,
    ModelProviderError,
    StructuredGenerationRequest,
)


class MockLLMProvider(LLMProvider):
    """返回预设文本的本地模型，不能用于真实业务处理。"""

    def __init__(
        self,
        response_content: str = "{}",
        *,
        model_name: str = "demo-mock-model",
        error_message: str | None = None,
    ) -> None:
        # 预设文本可以故意设置为非法 JSON，用于测试解析失败后的人工复核。
        self._response_content = response_content
        self._model_name = model_name
        self._error_message = error_message

        # 保存调用快照，测试可以检查 Agent 是否正确传递了 Schema 和提示词。
        self.requests: list[StructuredGenerationRequest] = []

    def generate_json(
        self, request: StructuredGenerationRequest
    ) -> LLMResponse:
        """记录请求并返回预设文本，或模拟模型提供方调用失败。"""

        # 使用深拷贝，避免调用方之后修改 request 影响已经记录的测试证据。
        self.requests.append(request.model_copy(deep=True))

        if self._error_message is not None:
            raise ModelProviderError(self._error_message)

        return LLMResponse(
            content=self._response_content,
            provider_name="mock",
            model_name=self._model_name,
        )
