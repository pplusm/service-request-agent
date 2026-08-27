"""验证第一步多模态视觉抽取的安全边界。"""

from __future__ import annotations

import base64
from pathlib import Path

from app.agent.triage_graph import TriageAgent
from app.llm.demo_provider import DemoLLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import (
    ReviewReason,
    ServiceRequestInput,
)
from app.vision.demo_provider import DemoVisionProvider
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProvider,
    VisionResponse,
)


class EchoImageVisionProvider(VisionProvider):
    """模拟错误服务方回显图片内容，用于验证脱敏边界。"""

    def generate_json(self, request: VisionExtractionRequest) -> VisionResponse:
        """故意返回非法内容，其中包含裸 Base64 与 data URL。"""

        image_data_url = (
            f"data:{request.image.media_type};base64,{request.image.data_base64}"
        )
        return VisionResponse(
            content=(
                "错误服务方回显了图片："
                f"{request.image.data_base64}；{image_data_url}"
            ),
            provider_name="echoing-test-vision",
            model_name="echoing-test-vision-v1",
        )


class GraphThatRaises:
    """模拟 LangGraph 在返回结果前整体中断。"""

    def __init__(self, error_message: str) -> None:
        self._error_message = error_message

    def invoke(self, _state: object) -> object:
        """抛出包含请求回显的异常，验证顶层兜底处理。"""

        raise RuntimeError(self._error_message)


def build_store(tmp_path: Path) -> ChromaKnowledgeStore:
    """创建隔离的临时知识库，避免测试修改开发机上的 Chroma 数据。"""

    project_root = Path(__file__).resolve().parents[2]
    store = ChromaKnowledgeStore(persist_directory=tmp_path / "chroma")
    store.index_directory(project_root / "data" / "scenic_service" / "knowledge")
    return store


def build_image_request(request_id: str = "vision_demo_001") -> ServiceRequestInput:
    """构造一份不含真实个人信息的演示图片请求。"""

    # 这里的字节只用于测试传输和摘要，不代表任何真实照片。
    demo_bytes = b"local-demo-image-bytes"
    return ServiceRequestInput(
        request_id=request_id,
        text="西门照明故障",
        image={
            "media_type": "image/png",
            "data_base64": base64.b64encode(demo_bytes).decode("ascii"),
            "filename": "demo.png",
        },
    )


def test_text_request_skips_vision_provider(tmp_path: Path) -> None:
    """没有图片时视觉提供方不能被调用，旧文本流程保持不变。"""

    vision_provider = DemoVisionProvider()
    agent = TriageAgent(
        knowledge_store=build_store(tmp_path),
        model_provider=DemoLLMProvider(),
        vision_provider=vision_provider,
    )

    result = agent.run(
        ServiceRequestInput(
            request_id="vision_demo_text_only",
            text="西门照明故障",
        )
    )

    assert vision_provider.requests == []
    assert result.image is None
    assert result.vision_observation is None
    assert result.diagnostics.vision_call_success is None
    assert result.diagnostics.vision_output_parse_success is None


def test_image_request_returns_structured_demo_observation(tmp_path: Path) -> None:
    """有图片时结果必须包含可记录的摘要和结构化视觉观察。"""

    vision_provider = DemoVisionProvider()
    agent = TriageAgent(
        knowledge_store=build_store(tmp_path),
        model_provider=DemoLLMProvider(),
        vision_provider=vision_provider,
    )

    result = agent.run(build_image_request())
    serialized = result.model_dump_json()

    assert len(vision_provider.requests) == 1
    assert result.image is not None
    assert result.image.media_type == "image/png"
    assert result.image.size_bytes == len(b"local-demo-image-bytes")
    assert result.vision_observation is not None
    assert result.vision_observation.is_demo_observation is True
    assert result.diagnostics.vision_call_success is True
    assert result.diagnostics.vision_output_parse_success is True
    # 图片二进制只在本次处理内存中使用，结果和历史 JSON 不得包含 base64 字段。
    assert "data_base64" not in serialized
    assert "local-demo-image-bytes" not in serialized


def test_vision_provider_error_requires_human_review(tmp_path: Path) -> None:
    """视觉服务调用失败时必须保留摘要并进入人工复核。"""

    agent = TriageAgent(
        knowledge_store=build_store(tmp_path),
        model_provider=DemoLLMProvider(),
        vision_provider=DemoVisionProvider(error_message="demo vision unavailable"),
    )

    result = agent.run(build_image_request("vision_demo_error"))

    assert result.image is not None
    assert result.vision_observation is None
    assert result.diagnostics.vision_call_success is False
    assert result.diagnostics.vision_output_parse_success is None
    assert ReviewReason.VISION_ERROR in result.review.reasons
    assert result.review.requires_human_review is True


def test_invalid_vision_output_requires_human_review(tmp_path: Path) -> None:
    """视觉模型返回非法 JSON 时必须保存受限原文并进入人工复核。"""

    agent = TriageAgent(
        knowledge_store=build_store(tmp_path),
        model_provider=DemoLLMProvider(),
        vision_provider=DemoVisionProvider(simulate_invalid_output=True),
    )

    result = agent.run(build_image_request("vision_demo_invalid"))

    assert result.diagnostics.vision_call_success is True
    assert result.diagnostics.vision_output_parse_success is False
    assert result.diagnostics.raw_vision_output == "这不是合法的视觉 JSON"
    assert ReviewReason.INVALID_VISION_OUTPUT in result.review.reasons
    assert result.review.requires_human_review is True


def test_vision_parse_failure_redacts_echoed_image_content(tmp_path: Path) -> None:
    """非法视觉输出不得把图片 Base64 或 data URL 写入结果。"""

    request = build_image_request("vision_demo_redaction")
    assert request.image is not None
    agent = TriageAgent(
        knowledge_store=build_store(tmp_path),
        model_provider=DemoLLMProvider(),
        vision_provider=EchoImageVisionProvider(),
    )

    result = agent.run(request)
    serialized = result.model_dump_json()

    assert request.image.data_base64 not in serialized
    assert f"data:{request.image.media_type};base64," not in serialized
    assert "[REDACTED_IMAGE_BASE64]" in serialized
    assert "[REDACTED_IMAGE_DATA_URL]" in serialized
    assert result.diagnostics.vision_output_parse_success is False
    assert ReviewReason.INVALID_VISION_OUTPUT in result.review.reasons


def test_graph_failure_with_image_keeps_metadata_and_requires_review(
    tmp_path: Path,
) -> None:
    """图级异常不能丢失图片摘要，且必须要求人工检查视觉流程。"""

    request = build_image_request("vision_demo_graph_error")
    assert request.image is not None
    agent = TriageAgent(
        knowledge_store=build_store(tmp_path),
        model_provider=DemoLLMProvider(),
    )
    # 这里只替换测试对象内部图，用来覆盖 Agent 的最外层异常兜底路径。
    agent._graph = GraphThatRaises(  # type: ignore[assignment]
        f"graph failed after receiving {request.image.data_base64}"
    )

    result = agent.run(request)
    serialized = result.model_dump_json()

    assert result.image == request.image.metadata()
    assert result.vision_observation is None
    assert result.diagnostics.vision_call_success is None
    assert ReviewReason.VISION_ERROR in result.review.reasons
    assert result.review.requires_human_review is True
    assert request.image.data_base64 not in serialized
