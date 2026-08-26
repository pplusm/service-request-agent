"""加载演示案例，运行 Agent，并生成经过 Pydantic 校验的 JSON 报告。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.agent.triage_graph import TriageAgent
from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationCaseSet,
    EvaluationReport,
    ProviderBehavior,
)
from app.llm.demo_provider import DemoLLMProvider
from app.llm.mock_provider import MockLLMProvider
from app.llm.provider import LLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.rules.scenic_service_config import load_scenic_service_configuration
from app.schemas.models import ServiceCaseResult


# 从本文件回到项目根目录，避免运行命令依赖当前 PowerShell 所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "scenarios" / "scenic_service" / "evaluation_cases.yaml"
)
DEFAULT_KNOWLEDGE_DIRECTORY = (
    PROJECT_ROOT / "data" / "scenic_service" / "knowledge"
)
DEFAULT_CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"


class EvaluationDatasetError(ValueError):
    """评估案例文件不存在、格式错误或未通过 Pydantic 校验时抛出。"""


def load_evaluation_case_set(path: Path = DEFAULT_CASES_PATH) -> EvaluationCaseSet:
    """从 YAML 加载评估案例，并拒绝任意未定义字段。"""

    case_path = Path(path)
    if not case_path.is_file():
        raise EvaluationDatasetError(
            f"evaluation cases file does not exist: {case_path}"
        )

    try:
        raw_content = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationDatasetError(
            f"failed to read evaluation cases from {case_path}: {error}"
        ) from error

    if not isinstance(raw_content, dict):
        raise EvaluationDatasetError(
            f"evaluation cases must contain a YAML object: {case_path}"
        )

    try:
        return EvaluationCaseSet.model_validate(raw_content)
    except ValidationError as error:
        raise EvaluationDatasetError(
            f"evaluation cases are invalid: {error}"
        ) from error


def run_evaluation(
    case_set: EvaluationCaseSet,
    *,
    knowledge_directory: Path = DEFAULT_KNOWLEDGE_DIRECTORY,
    persist_directory: Path = DEFAULT_CHROMA_DIRECTORY,
) -> EvaluationReport:
    """执行整组案例，返回可被 Pydantic 再次解析的汇总 JSON 模型。"""

    # 所有案例共享同一份本地知识库；案例间只替换演示模型的故障行为。
    store = ChromaKnowledgeStore(persist_directory=Path(persist_directory))
    store.index_directory(Path(knowledge_directory))
    agents = _build_agents(store=store)

    case_results: list[EvaluationCaseResult] = []
    for case in case_set.cases:
        actual = agents[case.provider_behavior].run(case.request)
        mismatches = compare_case_result(case=case, actual=actual)
        case_results.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                passed=not mismatches,
                mismatches=mismatches,
                actual=actual,
            )
        )

    passed_cases = sum(result.passed for result in case_results)
    review_required_cases = sum(
        result.actual.review.requires_human_review for result in case_results
    )
    review_required_cases_passed = sum(
        result.passed and result.actual.review.requires_human_review
        for result in case_results
    )
    return EvaluationReport(
        dataset_name=case_set.dataset_name,
        scenario=case_set.scenario,
        total_cases=len(case_results),
        passed_cases=passed_cases,
        failed_cases=len(case_results) - passed_cases,
        review_required_cases=review_required_cases,
        review_required_cases_passed=review_required_cases_passed,
        all_expectations_met=passed_cases == len(case_results),
        case_results=case_results,
    )


def _build_agents(*, store: ChromaKnowledgeStore) -> dict[ProviderBehavior, TriageAgent]:
    """为正常、解析失败和调用失败三种本地行为各建一个 Agent。"""

    configuration = load_scenic_service_configuration()
    providers: dict[ProviderBehavior, LLMProvider] = {
        ProviderBehavior.DEMO: DemoLLMProvider(configuration=configuration),
        ProviderBehavior.INVALID_OUTPUT: DemoLLMProvider(
            configuration=configuration,
            simulate_invalid_output=True,
        ),
        ProviderBehavior.PROVIDER_ERROR: MockLLMProvider(
            error_message="evaluation deliberately simulates a provider failure"
        ),
    }
    return {
        behavior: TriageAgent(
            knowledge_store=store,
            model_provider=provider,
            configuration=configuration,
        )
        for behavior, provider in providers.items()
    }


def compare_case_result(
    *,
    case: EvaluationCase,
    actual: ServiceCaseResult,
) -> list[str]:
    """只比较案例明确声明的演示预期，并返回易读的不匹配原因。"""

    expected = case.expected
    mismatches: list[str] = []

    if actual.request_id != case.case_id:
        mismatches.append(
            f"request_id: expected {case.case_id!r}, got {actual.request_id!r}"
        )
    if actual.classification.event_type != expected.event_type:
        mismatches.append(
            "classification.event_type: expected "
            f"{expected.event_type.value!r}, got "
            f"{actual.classification.event_type.value!r}"
        )
    if actual.risk.level != expected.risk_level:
        mismatches.append(
            f"risk.level: expected {expected.risk_level.value!r}, "
            f"got {actual.risk.level.value!r}"
        )
    if actual.review.requires_human_review != expected.requires_human_review:
        mismatches.append(
            "review.requires_human_review: expected "
            f"{expected.requires_human_review!r}, got "
            f"{actual.review.requires_human_review!r}"
        )

    actual_reasons = set(actual.review.reasons)
    missing_reasons = set(expected.minimum_review_reasons) - actual_reasons
    if missing_reasons:
        reason_values = sorted(reason.value for reason in missing_reasons)
        mismatches.append(f"review.reasons is missing {reason_values!r}")

    if actual.diagnostics.knowledge_hit != expected.knowledge_hit:
        mismatches.append(
            "diagnostics.knowledge_hit: expected "
            f"{expected.knowledge_hit!r}, got "
            f"{actual.diagnostics.knowledge_hit!r}"
        )
    if actual.diagnostics.model_call_success != expected.model_call_success:
        mismatches.append(
            "diagnostics.model_call_success: expected "
            f"{expected.model_call_success!r}, got "
            f"{actual.diagnostics.model_call_success!r}"
        )
    if (
        actual.diagnostics.model_output_parse_success
        != expected.model_output_parse_success
    ):
        mismatches.append(
            "diagnostics.model_output_parse_success: expected "
            f"{expected.model_output_parse_success!r}, got "
            f"{actual.diagnostics.model_output_parse_success!r}"
        )
    if len(actual.action_plan) != expected.action_plan_count:
        mismatches.append(
            "action_plan count: expected "
            f"{expected.action_plan_count}, got {len(actual.action_plan)}"
        )
    return mismatches


def main() -> None:
    """提供可直接在 PowerShell 运行的评估命令。"""

    if hasattr(sys.stdout, "reconfigure"):
        # Windows 终端默认编码可能不同，显式使用 UTF-8 保证中文 JSON 可读。
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="运行景区服务演示评估，并输出 Pydantic 校验后的 JSON 报告。"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="评估案例 YAML 路径。默认使用景区服务演示案例。",
    )
    parser.add_argument(
        "--chroma-directory",
        type=Path,
        default=DEFAULT_CHROMA_DIRECTORY,
        help="本地 ChromaDB 持久化目录。默认使用项目的 chroma_data。",
    )
    arguments = parser.parse_args()

    case_set = load_evaluation_case_set(arguments.cases)
    report = run_evaluation(
        case_set,
        persist_directory=arguments.chroma_directory,
    )
    print(report.model_dump_json(indent=2))

    # 让持续集成或 PowerShell 调用者能根据退出码判断是否全部通过。
    if not report.all_expectations_met:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
