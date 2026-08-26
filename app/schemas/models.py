from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 所有数据模型共用的严格校验规则：拒绝未定义字段，并自动去除文本首尾空格。
class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# 第一阶段只支持景区服务文本场景。枚举继承 str，输出 JSON 时会是普通字符串。
class ScenarioId(str, Enum):
    SCENIC_SERVICE = "scenic_service"


# 当前只接收文本；图片等多模态输入会在后续阶段单独加入。
class InputChannel(str, Enum):
    TEXT = "text"


# Agent 对诉求进行归类时只能从这些演示事件类型中选择，避免输出随意的类别名称。
class EventType(str, Enum):
    CROWD_CONGESTION = "crowd_congestion"
    FACILITY_SHORTAGE = "facility_shortage"
    VISITOR_HEALTH = "visitor_health"
    ENVIRONMENT_HYGIENE = "environment_hygiene"
    FACILITY_FAULT = "facility_fault"
    LOST_PERSON = "lost_person"
    SAFETY_INCIDENT = "safety_incident"
    OTHER_UNKNOWN = "other_unknown"


# 风险等级用于决定是否必须转交人工复核。
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNASSESSED = "unassessed"


class ServiceRequestInput(StrictSchema):
    """进入 Agent 的原始文本诉求。"""

    # 请求唯一编号由 API 或调用方提供，后续结果使用它关联同一个案件。
    request_id: str = Field(min_length=1, max_length=100)

    # 先固定为景区服务，避免第一阶段误接入未支持的业务场景。
    scenario: ScenarioId = ScenarioId.SCENIC_SERVICE

    # 原始诉求文本；长度限制用于防止异常大的输入。
    text: str = Field(min_length=1, max_length=2000)

    # 当前约定为文本输入，保留该字段是为了未来扩展其他输入渠道。
    source_channel: InputChannel = InputChannel.TEXT

    # 未提供接收时间时，自动记录当前 UTC 时间，便于审计和排查。
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ExtractedEntities(StrictSchema):
    """从原始文本中提取的结构化信息；无法确定时保持为 None。"""

    location: str | None = None
    facility_name: str | None = None
    visitor_condition: str | None = None
    estimated_affected_count: int | None = Field(default=None, ge=0)
    event_time_description: str | None = None

    # 例如 location、event_time；后续流程发现它们缺失时必须要求人工复核。
    missing_fields: list[str] = Field(default_factory=list)


class EventClassification(StrictSchema):
    """事件分类结果及其依据。"""

    event_type: EventType = EventType.OTHER_UNKNOWN

    # 置信度仅允许 0 到 1，不能把模型的自由文本直接当作数值使用。
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # 保存支持本次分类的原文片段，方便人工复核结果来源。
    evidence: list[str] = Field(default_factory=list)


class RiskAssessment(StrictSchema):
    """风险评估结果；默认 unassessed 表示尚不能安全地自动判断。"""

    level: RiskLevel = RiskLevel.UNASSESSED
    risk_factors: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=500)


# 需要人工复核的原因必须使用固定枚举，避免模型输出含义不清的自由文本。
class ReviewReason(str, Enum):
    HIGH_RISK = "high_risk"
    UNASSESSED_RISK = "unassessed_risk"
    MISSING_FIELDS = "missing_fields"
    KNOWLEDGE_NOT_FOUND = "knowledge_not_found"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    LLM_ERROR = "llm_error"
    RULE_CONFLICT = "rule_conflict"
    SENSITIVE_DATA = "sensitive_data"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_SCENARIO = "unsupported_scenario"


class KnowledgeReference(StrictSchema):
    """命中的演示知识来源，保证结果可以追溯到具体资料。"""

    # 例如 demo_facility_001；后续处置建议通过它关联知识来源。
    source_id: str = Field(min_length=1, max_length=100)

    # 来源标题和项目内文件路径用于人工查看原始演示资料。
    source_title: str = Field(min_length=1, max_length=200)
    source_path: str = Field(min_length=1, max_length=500)

    # 只保存本次判断实际使用的短摘录，而不是虚构完整服务规则。
    excerpt: str = Field(min_length=1, max_length=1000)
    relevance_score: float = Field(ge=0.0, le=1.0)

    # 第一阶段只能使用演示知识，Literal[True] 会拒绝 false。
    is_demo_source: Literal[True] = True


class KnowledgeSearchResult(StrictSchema):
    """RAG 子系统的检索结果，不代表完整的案件处置结论。"""

    query: str = Field(min_length=1, max_length=2000)
    knowledge_hit: bool
    knowledge_references: list[KnowledgeReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_knowledge_hit(self) -> "KnowledgeSearchResult":
        """确保“是否命中”的标记与实际引用列表一致。"""

        if self.knowledge_hit != bool(self.knowledge_references):
            raise ValueError(
                "knowledge_hit must match whether knowledge_references is empty"
            )
        return self


class ActionPlanItem(StrictSchema):
    """基于演示知识生成的建议动作，不代表真实景区处置指令。"""

    step: int = Field(ge=1)
    suggested_action: str = Field(min_length=1, max_length=500)

    # 每条建议都必须在后续校验中关联至少一个知识来源。
    knowledge_source_ids: list[str] = Field(min_length=1)

    # 明确标记为演示建议，避免被误认为真实业务规则。
    is_demo_action: Literal[True] = True


class ReviewDecision(StrictSchema):
    """是否必须交由人工确认，以及触发人工复核的原因。"""

    requires_human_review: bool = False
    reasons: list[ReviewReason] = Field(default_factory=list)
    review_note: str = Field(default="", max_length=500)


class Diagnostics(StrictSchema):
    """供系统审计和排错使用的信息，不作为真实业务结论。"""

    # 知识库是否命中；未命中时后续校验会强制人工复核。
    knowledge_hit: bool = False

    # None 表示本次未调用模型；False 表示模型服务调用失败。
    model_call_success: bool | None = None

    # None 表示没有模型输出可解析；False 表示原始输出未通过 Pydantic 校验。
    model_output_parse_success: bool | None = None

    # 只有解析失败时才保留原始输出，供人工定位问题。
    raw_model_output: str | None = Field(default=None, max_length=5000)

    # 记录系统错误摘要，例如模型调用失败或规则冲突。
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_model_diagnostics(self) -> "Diagnostics":
        """保证模型调用、解析状态与原始输出之间没有矛盾。"""

        if (
            self.model_output_parse_success is not None
            and self.model_call_success is not True
        ):
            raise ValueError(
                "model_output_parse_success requires a successful model call"
            )

        if (
            self.raw_model_output is not None
            and self.model_output_parse_success is not False
        ):
            raise ValueError(
                "raw_model_output is only allowed when model parsing fails"
            )

        if (
            self.model_output_parse_success is False
            and self.raw_model_output is None
        ):
            raise ValueError(
                "raw_model_output is required when model parsing fails"
            )
        return self


class ServiceCaseResult(StrictSchema):
    """Agent 返回的完整案件结果，也是后续 API 必须输出的 JSON 结构。"""

    # 用输入时的 request_id 关联本次处理结果，不直接回传可能含敏感内容的原文。
    request_id: str = Field(min_length=1, max_length=100)
    scenario: ScenarioId = ScenarioId.SCENIC_SERVICE

    entities: ExtractedEntities
    classification: EventClassification
    risk: RiskAssessment

    # 命中的演示知识来源；为空表示知识库未命中。
    knowledge_references: list[KnowledgeReference] = Field(default_factory=list)

    # 仅能给出有知识来源支持的演示建议，不能虚构真实处置规则。
    action_plan: list[ActionPlanItem] = Field(default_factory=list)

    review: ReviewDecision
    diagnostics: Diagnostics

    @model_validator(mode="after")
    def enforce_human_review_rules(self) -> "ServiceCaseResult":
        """检查必须人工复核的情形，阻止不安全的结果通过校验。"""

        required_reasons: set[ReviewReason] = set()

        # 高风险和暂时无法评估风险的案件，都不能自动结案。
        if self.risk.level == RiskLevel.HIGH:
            required_reasons.add(ReviewReason.HIGH_RISK)
        elif self.risk.level == RiskLevel.UNASSESSED:
            required_reasons.add(ReviewReason.UNASSESSED_RISK)

        # 文本中缺少关键字段时，需要人工补充或确认。
        if self.entities.missing_fields:
            required_reasons.add(ReviewReason.MISSING_FIELDS)

        # 没有知识引用或诊断结果表明未命中知识库时，不能生成自动处置结论。
        if not self.knowledge_references or not self.diagnostics.knowledge_hit:
            required_reasons.add(ReviewReason.KNOWLEDGE_NOT_FOUND)

        # 模型服务调用失败时，必须交由人工判断，不能伪装成正常分类结果。
        if self.diagnostics.model_call_success is False:
            required_reasons.add(ReviewReason.LLM_ERROR)

        # 模型返回内容无法解析为 JSON 时，必须交给人工查看原始输出。
        if self.diagnostics.model_output_parse_success is False:
            required_reasons.add(ReviewReason.INVALID_MODEL_OUTPUT)

            if self.diagnostics.raw_model_output is None:
                raise ValueError(
                    "raw_model_output is required when model parsing fails"
                )

        # 0.60 是本项目的演示性保守阈值，不代表模型真实准确率。
        if self.classification.confidence < 0.60:
            required_reasons.add(ReviewReason.LOW_CONFIDENCE)

        # 诊断中的知识命中状态必须与知识引用列表一致。
        if self.diagnostics.knowledge_hit != bool(self.knowledge_references):
            raise ValueError(
                "knowledge_hit must match whether knowledge_references is empty"
            )

        # 知识未命中时，不允许保留任何看似有依据的处置建议。
        if not self.knowledge_references and self.action_plan:
            raise ValueError(
                "action_plan must be empty when knowledge is not found"
            )

        # 每条演示建议引用的 source_id 都必须真实存在于本次命中的知识列表。
        known_source_ids = {
            reference.source_id for reference in self.knowledge_references
        }
        for item in self.action_plan:
            unknown_source_ids = set(item.knowledge_source_ids) - known_source_ids
            if unknown_source_ids:
                raise ValueError(
                    "action_plan contains unknown knowledge source ids: "
                    f"{sorted(unknown_source_ids)}"
                )

        # 只要命中任一风险条件，人工复核开关和对应原因就缺一不可。
        if required_reasons:
            if not self.review.requires_human_review:
                raise ValueError(
                    "human review is required for this case"
                )

            missing_reasons = required_reasons - set(self.review.reasons)
            if missing_reasons:
                missing_values = sorted(
                    reason.value for reason in missing_reasons
                )
                raise ValueError(
                    "review reasons are missing required values: "
                    f"{missing_values}"
                )

        return self
