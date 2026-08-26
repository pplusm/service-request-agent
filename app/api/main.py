"""FastAPI 应用入口：启动时初始化本地演示知识库和 Agent。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from app.agent.triage_graph import TriageAgent
from app.api.schemas import TriageApiRequest
from app.api.service import TriageApiService, build_request_validation_error_result
from app.llm.demo_provider import DemoLLMProvider
from app.llm.provider import LLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import ServiceCaseResult


# 固定从 app/api/main.py 回到项目根目录，避免依赖启动命令所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "data" / "scenic_service" / "knowledge"
DEFAULT_CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"


def create_app(
    *,
    knowledge_directory: Path = DEFAULT_KNOWLEDGE_DIRECTORY,
    chroma_directory: Path = DEFAULT_CHROMA_DIRECTORY,
    provider_factory: Callable[[], LLMProvider] = DemoLLMProvider,
) -> FastAPI:
    """创建可测试的 FastAPI 应用；测试可注入独立的临时知识库目录。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 知识文件只在服务启动时入库一次，每次请求无需重复建立向量库。
        knowledge_store = ChromaKnowledgeStore(
            persist_directory=chroma_directory,
        )
        knowledge_store.index_directory(knowledge_directory)
        app.state.triage_service = TriageApiService(
            TriageAgent(
                knowledge_store=knowledge_store,
                model_provider=provider_factory(),
            )
        )
        yield

    app = FastAPI(
        title="景区服务诉求分诊与处置 Agent",
        description="仅使用本地演示资料的文本诉求分诊接口。",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        """JSON 格式错误或字段类型错误也返回 Pydantic 校验过的复核结果。"""

        result = build_request_validation_error_result(error.errors())
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            content=result.model_dump(mode="json"),
        )

    @app.post(
        "/api/v1/triage",
        response_model=ServiceCaseResult,
        summary="提交一条景区服务文本诉求",
    )
    def triage(
        request: TriageApiRequest | None = Body(default=None),
    ) -> ServiceCaseResult:
        """调用 Agent 并返回唯一的 Pydantic 案件结果结构。"""

        return app.state.triage_service.triage(request)

    return app


# Uvicorn 使用 app.api.main:app 启动时会读取这个应用实例。
app = create_app()
