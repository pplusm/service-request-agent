"""定义评估案例、单案结果和汇总报告的 Pydantic 数据模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import re

from pydantic import Field, model_validator

from app.schemas.models import (
    EventType,
    MultimodalFusionStatus,
    ReviewReason,
    RiskLevel,
    ScenarioId,
    ServiceCaseResult,
    ServiceRequestInput,
    StrictSchema,
)


class ProviderBehavior(str, Enum):
    """评估时使用的本地模型行为，不会调用任何外部 API。"""

    DEMO = "demo"
    INVALID_OUTPUT = "invalid_output"
    PROVIDER_ERROR = "provider_error"


class EvaluationCaseCategory(str, Enum):
    """评测案例所属的演示类别，不代表真实业务的案件分类。"""

    NORMAL = "normal"
    MISSING_FIELDS = "missing_fields"
    HIGH_RISK = "high_risk"
    KNOWLEDGE_NOT_FOUND = "knowledge_not_found"
    ADVERSARIAL = "adversarial"
    PROVIDER_FAILURE = "provider_failure"
    MULTIMODAL = "multimodal"


class VisionBehavior(str, Enum):
    """评测专用的确定性视觉观察类型，不会读取或识别图片像素。"""

    DEMO = "demo"
    LIGHTING_FAULT = "lighting_fault"
    LIGHTING_NORMAL = "lighting_normal"
    LOW_CONFIDENCE = "low_confidence"


# 图文评测只能引用 data/scenic_service/evaluation_images 中的单个文件名。
# 这样案例 YAML 不保存 Base64，也不能借文件名读取项目目录之外的内容。
_IMAGE_FIXTURE_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:png|jpe?g|webp)"
)


class ExpectedCaseOutcome(StrictSchema):
    """一条演示案例需要验证的最小结果集合。"""

    # 这些标签只用于验证本仓库的演示规则，不能当作真实业务准确率。
    event_type: EventType
    risk_level: RiskLevel
    requires_human_review: bool
    minimum_review_reasons: list[ReviewReason] = Field(default_factory=list)
    knowledge_hit: bool
    model_call_success: bool | None
    model_output_parse_success: bool | None
    action_plan_count: int = Field(ge=0, le=10)
    # 无图片时三个视觉预期都必须为 None；有图片时由 EvaluationCase 再检查组合。
    vision_call_success: bool | None = None
    vision_output_parse_success: bool | None = None
    multimodal_fusion_status: MultimodalFusionStatus | None = None

    @model_validator(mode="after")
    def validate_review_expectation(self) -> "ExpectedCaseOutcome":
        """不允许为无需人工复核的案例声明必须出现复核原因。"""

        if not self.requires_human_review and self.minimum_review_reasons:
            raise ValueError(
                "minimum_review_reasons require requires_human_review to be true"
            )
        return self


class EvaluationCase(StrictSchema):
    """一条可重复执行的演示测试案例。"""

    case_id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    category: EvaluationCaseCategory = EvaluationCaseCategory.NORMAL
    provider_behavior: ProviderBehavior = ProviderBehavior.DEMO
    # 图片内容不放在 YAML 中；运行器会从固定夹具目录读取并临时编码。
    image_fixture: str | None = Field(default=None, max_length=120)
    vision_behavior: VisionBehavior | None = None
    request: ServiceRequestInput
    expected: ExpectedCaseOutcome

    @model_validator(mode="after")
    def validate_case_contract(self) -> "EvaluationCase":
        """校验案件编号和图文夹具约束，避免评测数据绕过安全边界。"""

        if self.request.request_id != self.case_id:
            raise ValueError("request.request_id must match case_id")

        expected = self.expected
        has_vision_expectation = any(
            value is not None
            for value in (
                expected.vision_call_success,
                expected.vision_output_parse_success,
                expected.multimodal_fusion_status,
            )
        )

        if self.image_fixture is None:
            # YAML 不允许内嵌图片 Base64，文本案例也不应声明视觉行为或视觉预期。
            if self.request.image is not None:
                raise ValueError("evaluation YAML must not embed image data")
            if self.vision_behavior is not None or has_vision_expectation:
                raise ValueError(
                    "vision behavior and expectations require image_fixture"
                )
            if self.category == EvaluationCaseCategory.MULTIMODAL:
                raise ValueError("multimodal category requires image_fixture")
            return self

        if not _IMAGE_FIXTURE_NAME_PATTERN.fullmatch(self.image_fixture):
            raise ValueError(
                "image_fixture must be a single supported image filename"
            )
        if self.request.image is not None:
            raise ValueError("evaluation YAML must not embed image data")
        if self.category != EvaluationCaseCategory.MULTIMODAL:
            raise ValueError("image_fixture is only allowed for multimodal cases")
        if self.vision_behavior is None:
            raise ValueError("image_fixture requires vision_behavior")
        if expected.vision_call_success is None:
            raise ValueError("image_fixture requires vision_call_success expectation")

        if expected.vision_call_success is False:
            if (
                expected.vision_output_parse_success is not None
                or expected.multimodal_fusion_status is not None
            ):
                raise ValueError(
                    "failed vision call must not expect parsing or fusion"
                )
        elif expected.vision_output_parse_success is None:
            raise ValueError(
                "successful vision call requires parsing expectation"
            )
        elif expected.vision_output_parse_success is False:
            if expected.multimodal_fusion_status is not None:
                raise ValueError(
                    "failed vision parsing must not expect fusion status"
                )
        elif expected.multimodal_fusion_status is None:
            raise ValueError(
                "successful vision parsing requires fusion status expectation"
            )
        return self


class EvaluationCaseSet(StrictSchema):
    """一个场景的一组演示评估案例。"""

    dataset_name: str = Field(min_length=1, max_length=100)
    scenario: ScenarioId
    cases: list[EvaluationCase] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_cases(self) -> "EvaluationCaseSet":
        """拒绝重复案例编号和混入其他场景的输入。"""

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        if any(case.request.scenario != self.scenario for case in self.cases):
            raise ValueError("every evaluation case must use the dataset scenario")
        return self


class EvaluationCaseResult(StrictSchema):
    """一条案例的实际 Agent 输出与预期比对结果。"""

    case_id: str = Field(min_length=1, max_length=100)
    passed: bool
    mismatches: list[str] = Field(default_factory=list)
    actual: ServiceCaseResult

    @model_validator(mode="after")
    def validate_pass_status(self) -> "EvaluationCaseResult":
        """通过状态必须与是否发现不匹配严格一致。"""

        if self.passed != (not self.mismatches):
            raise ValueError("passed must match whether mismatches is empty")
        return self


class EvaluationReport(StrictSchema):
    """一次评估运行的完整、可序列化报告。"""

    dataset_name: str = Field(min_length=1, max_length=100)
    scenario: ScenarioId
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    review_required_cases: int = Field(ge=0)
    review_required_cases_passed: int = Field(ge=0)
    all_expectations_met: bool
    case_results: list[EvaluationCaseResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_summary_counts(self) -> "EvaluationReport":
        """阻止摘要数字与逐案结果脱节。"""

        expected_total = len(self.case_results)
        expected_passed = sum(result.passed for result in self.case_results)
        expected_review_cases = sum(
            result.actual.review.requires_human_review
            for result in self.case_results
        )
        expected_review_passed = sum(
            result.passed and result.actual.review.requires_human_review
            for result in self.case_results
        )

        if self.total_cases != expected_total:
            raise ValueError("total_cases must match case_results")
        if self.passed_cases != expected_passed:
            raise ValueError("passed_cases must match case_results")
        if self.failed_cases != self.total_cases - self.passed_cases:
            raise ValueError("failed_cases must match total_cases minus passed_cases")
        if self.review_required_cases != expected_review_cases:
            raise ValueError("review_required_cases must match actual results")
        if self.review_required_cases_passed != expected_review_passed:
            raise ValueError(
                "review_required_cases_passed must match actual results"
            )
        if self.all_expectations_met != (self.failed_cases == 0):
            raise ValueError("all_expectations_met must match failed_cases")
        return self
