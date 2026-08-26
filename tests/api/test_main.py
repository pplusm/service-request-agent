"""验证 FastAPI 接口始终返回经过 Pydantic 校验的案件 JSON。"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.schemas.models import ReviewReason, RiskLevel, ServiceCaseResult


def build_client(tmp_path: Path) -> TestClient:
    """创建使用临时 ChromaDB 目录的测试客户端，避免污染本地演示数据。"""

    project_root = Path(__file__).resolve().parents[2]
    app = create_app(
        knowledge_directory=project_root / "data" / "scenic_service" / "knowledge",
        chroma_directory=tmp_path / "chroma",
    )
    return TestClient(app)


def parse_case_result(response_json: dict[str, object]) -> ServiceCaseResult:
    """确认 HTTP 响应与核心 Pydantic 输出模型完全兼容。"""

    return ServiceCaseResult.model_validate(response_json)


def test_triage_api_returns_a_demo_action_for_supported_request(
    tmp_path: Path,
) -> None:
    """正常的演示设施问题应通过 API 返回带来源引用的建议。"""

    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            json={
                "request_id": "api_demo_001",
                "text": "西门照明故障",
            },
        )

    result = parse_case_result(response.json())
    assert response.status_code == 200
    assert result.review.requires_human_review is False
    assert result.entities.facility_name == "照明设施"
    assert len(result.action_plan) == 1


def test_triage_api_turns_missing_text_into_human_review(tmp_path: Path) -> None:
    """请求字段缺失时，不应由 FastAPI 默认报错绕过人工复核规则。"""

    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            json={"request_id": "api_demo_002"},
        )

    result = parse_case_result(response.json())
    assert response.status_code == 200
    assert result.request_id == "api_demo_002"
    assert result.review.requires_human_review is True
    assert ReviewReason.MISSING_FIELDS in result.review.reasons
    assert result.diagnostics.model_call_success is None


def test_triage_api_skips_model_when_knowledge_is_not_found(tmp_path: Path) -> None:
    """无关文本未命中本地资料时，接口必须直接转人工复核。"""

    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            json={
                "request_id": "api_demo_003",
                "text": "航班改签咨询",
            },
        )

    result = parse_case_result(response.json())
    assert response.status_code == 200
    assert ReviewReason.KNOWLEDGE_NOT_FOUND in result.review.reasons
    assert result.diagnostics.model_call_success is None


def test_triage_api_keeps_high_risk_request_for_human_review(
    tmp_path: Path,
) -> None:
    """高风险文本即使有演示知识来源，也不能自动给出处置建议。"""

    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            json={
                "request_id": "api_demo_004",
                "text": "东门卫生间没水，有游客摔倒。",
            },
        )

    result = parse_case_result(response.json())
    assert response.status_code == 200
    assert result.risk.level == RiskLevel.HIGH
    assert ReviewReason.HIGH_RISK in result.review.reasons
    assert result.action_plan == []


def test_triage_api_returns_a_case_result_for_malformed_json(
    tmp_path: Path,
) -> None:
    """无法解析的 JSON 也必须返回可校验的人工复核结果。"""

    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/triage",
            content='{"request_id":',
            headers={"content-type": "application/json"},
        )

    result = parse_case_result(response.json())
    assert response.status_code == 422
    assert result.review.requires_human_review is True
    assert ReviewReason.MISSING_FIELDS in result.review.reasons
    assert result.diagnostics.model_call_success is None
