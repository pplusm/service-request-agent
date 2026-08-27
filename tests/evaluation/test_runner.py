"""验证演示评估案例、比对逻辑和 JSON 报告都可重复运行。"""

from collections import Counter
from pathlib import Path

from app.evaluation.models import (
    ExpectedCaseOutcome,
    EvaluationCase,
    EvaluationCaseCategory,
)
from app.evaluation.runner import (
    compare_case_result,
    load_evaluation_case_set,
    run_evaluation,
)
from app.llm.demo_provider import DemoLLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import EventType, RiskLevel, ServiceRequestInput
from app.agent.triage_graph import TriageAgent


def test_default_evaluation_cases_all_pass_and_report_roundtrips(
    tmp_path: Path,
) -> None:
    """基线案例应全部通过，且报告 JSON 能再次通过 Pydantic 校验。"""

    case_set = load_evaluation_case_set()
    report = run_evaluation(
        case_set,
        persist_directory=tmp_path / "chroma",
    )

    assert report.total_cases == 84
    assert report.passed_cases == 84
    assert report.failed_cases == 0
    assert report.review_required_cases == 55
    assert report.review_required_cases_passed == 55
    assert report.all_expectations_met is True
    assert report.model_validate_json(report.model_dump_json()) == report

    # 80 条文本和 4 条图文的类别分布是这份基线数据的范围契约。
    assert Counter(case.category.value for case in case_set.cases) == {
        "normal": 28,
        "missing_fields": 15,
        "high_risk": 15,
        "knowledge_not_found": 10,
        "adversarial": 10,
        "provider_failure": 2,
        "multimodal": 4,
    }

    # 图文样例只验证流程契约：其中的视觉观察由确定性模拟值提供，不能视为识图准确率。
    case_by_id = {case.case_id: case for case in case_set.cases}
    multimodal_results = [
        result
        for result in report.case_results
        if case_by_id[result.case_id].category == EvaluationCaseCategory.MULTIMODAL
    ]
    assert len(multimodal_results) == 4
    assert {
        result.actual.multimodal_fusion.status.value
        for result in multimodal_results
        if result.actual.multimodal_fusion is not None
    } == {
        "not_assessed",
        "consistent",
        "conflict",
        "insufficient_evidence",
    }


def test_compare_case_result_exposes_an_unmet_expectation(tmp_path: Path) -> None:
    """预期写错时，评估报告必须明确指出不匹配字段而不是误报通过。"""

    project_root = Path(__file__).resolve().parents[2]
    knowledge_directory = project_root / "data" / "scenic_service" / "knowledge"
    store = ChromaKnowledgeStore(persist_directory=tmp_path / "chroma")
    store.index_directory(knowledge_directory)
    agent = TriageAgent(
        knowledge_store=store,
        model_provider=DemoLLMProvider(),
    )
    case = EvaluationCase(
        case_id="eval_deliberate_mismatch",
        description="故意写错风险等级，验证评估器会报告差异。",
        request=ServiceRequestInput(
            request_id="eval_deliberate_mismatch",
            text="东门附近卫生间没水。",
        ),
        expected=ExpectedCaseOutcome(
            event_type=EventType.FACILITY_FAULT,
            risk_level=RiskLevel.HIGH,
            requires_human_review=False,
            knowledge_hit=True,
            model_call_success=True,
            model_output_parse_success=True,
            action_plan_count=1,
        ),
    )

    mismatches = compare_case_result(case=case, actual=agent.run(case.request))

    assert mismatches == ["risk.level: expected 'high', got 'low'"]
