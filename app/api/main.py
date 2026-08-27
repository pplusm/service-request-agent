"""FastAPI 应用入口：启动时初始化本地演示知识库和 Agent。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from app.agent.triage_graph import TriageAgent
from app.case_history.models import (
    CaseHistoryResponse,
    HumanReviewQueueResponse,
)
from app.case_history.repository import (
    CaseHistoryStorageError,
    LocalCaseHistoryRepository,
)
from app.api.schemas import TriageApiRequest
from app.api.service import TriageApiService, build_request_validation_error_result
from app.llm.factory import build_model_provider_from_environment
from app.llm.provider import LLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import ServiceCaseResult
from app.vision.factory import build_vision_provider_from_environment
from app.vision.provider import VisionProvider


# 固定从 app/api/main.py 回到项目根目录，避免依赖启动命令所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "data" / "scenic_service" / "knowledge"
DEFAULT_CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"
# 容器运行时可用环境变量把本地案件历史放到独立数据卷；本地开发仍使用原来的默认位置。
DEFAULT_CASE_HISTORY_DATABASE = Path(
    os.getenv(
        "CASE_HISTORY_DATABASE",
        str(PROJECT_ROOT / "data" / "case_history.sqlite3"),
    )
)


def create_app(
    *,
    knowledge_directory: Path = DEFAULT_KNOWLEDGE_DIRECTORY,
    chroma_directory: Path = DEFAULT_CHROMA_DIRECTORY,
    case_history_database: Path = DEFAULT_CASE_HISTORY_DATABASE,
    provider_factory: Callable[[], LLMProvider] = (
        build_model_provider_from_environment
    ),
    vision_provider_factory: Callable[[], VisionProvider] = (
        build_vision_provider_from_environment
    ),
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
                vision_provider=vision_provider_factory(),
            )
        )
        # 历史库只保存通过 Pydantic 校验的结果 JSON，不单独保存原始请求体。
        app.state.case_history_repository = LocalCaseHistoryRepository(
            database_path=case_history_database,
        )
        yield

    app = FastAPI(
        title="景区服务诉求分诊与处置 Agent",
        description="使用本地演示资料的文本和可选图片诉求分诊接口。",
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
        # 即使请求 JSON 本身不合法，生成的保守复核结果也应留在本地队列中。
        _request.app.state.case_history_repository.save(result)
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

        result = app.state.triage_service.triage(request)
        app.state.case_history_repository.save(result)
        return result

    @app.get(
        "/api/v1/case-history",
        response_model=CaseHistoryResponse,
        summary="查看本地案件历史",
    )
    def case_history() -> CaseHistoryResponse:
        """返回最近 100 条本地演示案件；读取异常也使用 Pydantic JSON 表达。"""

        try:
            records = app.state.case_history_repository.list_recent()
        except CaseHistoryStorageError as error:
            return CaseHistoryResponse(storage_error=str(error))
        return CaseHistoryResponse(records=records)

    @app.get(
        "/api/v1/review-queue",
        response_model=HumanReviewQueueResponse,
        summary="查看待人工复核的本地案件",
    )
    def review_queue() -> HumanReviewQueueResponse:
        """只返回安全规则已标记为必须人工复核的最近 100 条案件。"""

        try:
            records = app.state.case_history_repository.list_pending_human_review()
        except CaseHistoryStorageError as error:
            return HumanReviewQueueResponse(storage_error=str(error))
        return HumanReviewQueueResponse(records=records)

    return app


# Uvicorn 使用 app.api.main:app 启动时会读取这个应用实例。
app = create_app()
