"""验证本地终端演示脚本中模拟模型的可控行为。"""

from pathlib import Path

from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import ReviewReason, RiskLevel, ServiceCaseResult
from scenarios.scenic_service.run_demo_agent import run_agent


def build_store(tmp_path: Path) -> ChromaKnowledgeStore:
    """使用临时目录创建只含演示资料的知识库。"""

    project_root = Path(__file__).resolve().parents[2]
    knowledge_directory = project_root / "data" / "scenic_service" / "knowledge"
    store = ChromaKnowledgeStore(persist_directory=tmp_path / "chroma")
    store.index_directory(knowledge_directory)
    return store


def run_demo(
    store: ChromaKnowledgeStore,
    text: str,
    *,
    simulate_invalid_output: bool = False,
) -> ServiceCaseResult:
    """通过终端脚本的公开函数运行一次完整演示。"""

    return run_agent(
        store,
        request_id="script_demo_001",
        text=text,
        simulate_invalid_output=simulate_invalid_output,
    )


def test_demo_mock_supports_lighting_fault_with_a_location(tmp_path: Path) -> None:
    """演示资料列出的照明故障应产生可用的演示性建议。"""

    result = run_demo(build_store(tmp_path), "西门照明故障")

    assert result.entities.location == "西门"
    assert result.entities.facility_name == "照明设施"
    assert result.review.requires_human_review is False
    assert len(result.action_plan) == 1


def test_demo_mock_requires_review_when_location_is_missing(tmp_path: Path) -> None:
    """支持的设施问题缺少地点时，不能自动给出演示建议。"""

    result = run_demo(build_store(tmp_path), "卫生间没水")

    assert result.review.requires_human_review is True
    assert ReviewReason.MISSING_FIELDS in result.review.reasons
    assert result.action_plan == []


def test_demo_mock_keeps_high_risk_cases_for_human_review(tmp_path: Path) -> None:
    """高风险提示必须保留人工复核，且不能生成自动动作建议。"""

    result = run_demo(
        build_store(tmp_path),
        "东门卫生间没水，有游客摔倒。",
    )

    assert result.risk.level == RiskLevel.HIGH
    assert result.review.requires_human_review is True
    assert ReviewReason.HIGH_RISK in result.review.reasons
    assert result.action_plan == []


def test_demo_mock_only_simulates_invalid_output_when_requested(
    tmp_path: Path,
) -> None:
    """模型解析失败必须由显式命令行开关触发，而不是普通输入的默认结果。"""

    result = run_demo(
        build_store(tmp_path),
        "西门照明故障",
        simulate_invalid_output=True,
    )

    assert result.review.requires_human_review is True
    assert ReviewReason.INVALID_MODEL_OUTPUT in result.review.reasons
    assert result.diagnostics.model_output_parse_success is False
