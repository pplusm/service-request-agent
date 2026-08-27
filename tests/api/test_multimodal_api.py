"""验证图片请求的 FastAPI 输入和本地历史安全行为。"""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.case_history.models import CaseHistoryResponse
from app.schemas.models import ReviewReason, ServiceCaseResult


def build_client(tmp_path: Path) -> TestClient:
    """创建使用临时 ChromaDB 和 SQLite 的测试客户端。"""

    project_root = Path(__file__).resolve().parents[2]
    app = create_app(
        knowledge_directory=project_root / "data" / "scenic_service" / "knowledge",
        chroma_directory=tmp_path / "chroma",
        case_history_database=tmp_path / "case_history.sqlite3",
    )
    return TestClient(app)


def test_api_accepts_optional_image_and_returns_structured_observation(
    tmp_path: Path,
) -> None:
    """合法图片请求应返回图片摘要和结构化视觉观察。"""

    encoded = base64.b64encode(b"api-demo-image").decode("ascii")
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            json={
                "request_id": "api_vision_001",
                "text": "西门照明故障",
                "image": {
                    "media_type": "image/png",
                    "data_base64": encoded,
                    "filename": "demo.png",
                },
            },
        )

        result = ServiceCaseResult.model_validate(response.json())
        history = CaseHistoryResponse.model_validate(
            client.get("/api/v1/case-history").json()
        )

    assert response.status_code == 200
    assert result.image is not None
    assert result.vision_observation is not None
    assert result.diagnostics.vision_call_success is True
    assert result.diagnostics.vision_output_parse_success is True
    # 历史记录也只能保存摘要，不能保存原始图片内容。
    history_json = history.model_dump_json()
    assert "data_base64" not in history_json
    assert encoded not in history_json


def test_api_invalid_image_is_a_valid_human_review_result(tmp_path: Path) -> None:
    """非法 base64 图片不能绕过统一的 Pydantic 人工复核结果。"""

    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            json={
                "request_id": "api_vision_invalid",
                "text": "西门照明故障",
                "image": {
                    "media_type": "image/png",
                    "data_base64": "not-base64!",
                },
            },
        )

    result = ServiceCaseResult.model_validate(response.json())
    assert response.status_code == 422
    assert result.review.requires_human_review is True
    assert ReviewReason.INVALID_IMAGE in result.review.reasons

