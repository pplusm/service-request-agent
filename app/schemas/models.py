import base64
import binascii
import hashlib
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


# 输入渠道先支持纯文本和“文本 + 图片”；图片不能绕过文本场景约束。
class InputChannel(str, Enum):
    TEXT = "text"
    TEXT_IMAGE = "text_image"


# 图片只作为演示输入在内存中短暂传递，不写入案件历史数据库。
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_IMAGE_BASE64_LENGTH = 7_000_000
ImageMediaType = Literal["image/jpeg", "image/png", "image/webp"]


class ImageMetadata(StrictSchema):
    """可安全记录的图片元数据，不包含图片二进制内容。"""

    # 只允许常见的三种图片格式，避免把任意文件伪装成图片传入视觉模型。
    media_type: ImageMediaType
    size_bytes: int = Field(ge=1, le=_MAX_IMAGE_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str | None = Field(default=None, max_length=255)


class ImageAttachment(StrictSchema):
    """一次请求中的图片内容及其校验后的摘要。"""

    media_type: ImageMediaType
    # API 使用 base64 传输图片；长度上限约对应 5 MiB 的原始内容。
    data_base64: str = Field(
        min_length=1,
        max_length=_MAX_IMAGE_BASE64_LENGTH,
    )
    filename: str | None = Field(default=None, max_length=255)
    # 以下两个字段由校验器计算，也允许调用方提供后进行一致性核对。
    size_bytes: int | None = Field(default=None, ge=1, le=_MAX_IMAGE_BYTES)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def normalize_base64_aliases(cls, value: object) -> object:
        """兼容 image_base64 字段和浏览器常见的 data URL 写法。"""

        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if "image_base64" in normalized:
            # 同时传两个字段容易造成审计摘要与实际内容不一致，因此拒绝。
            if "data_base64" in normalized:
                raise ValueError(
                    "只允许提供 data_base64 或 image_base64 其中一个字段"
                )
            normalized["data_base64"] = normalized.pop("image_base64")

        raw_value = normalized.get("data_base64")
        if not isinstance(raw_value, str):
            return normalized

        raw_value = raw_value.strip()
        if raw_value.lower().startswith("data:"):
            # 接受 data:image/png;base64,...，但仍以独立 media_type 字段为准校验。
            header, separator, payload = raw_value.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ValueError("图片 data URL 必须包含 ;base64 标记")
            data_url_media_type = header[5:].split(";", 1)[0].lower()
            declared_media_type = normalized.get("media_type")
            if (
                declared_media_type is not None
                and declared_media_type != data_url_media_type
            ):
                raise ValueError("data URL 的媒体类型与 media_type 不一致")
            normalized.setdefault("media_type", data_url_media_type)
            raw_value = payload

        # 去掉换行和空格，便于直接粘贴经过换行的 base64 文本。
        normalized["data_base64"] = "".join(raw_value.split())
        return normalized

    @model_validator(mode="after")
    def validate_and_hash_content(self) -> "ImageAttachment":
        """严格解码图片内容，并计算不可逆的审计摘要。"""

        try:
            decoded = base64.b64decode(self.data_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("data_base64 不是有效的 base64 内容") from error

        if not decoded:
            raise ValueError("图片内容不能为空")
        if len(decoded) > _MAX_IMAGE_BYTES:
            raise ValueError("图片大小不能超过 5 MiB")
        if self.size_bytes is not None and self.size_bytes != len(decoded):
            raise ValueError("size_bytes 与图片实际大小不一致")

        digest = hashlib.sha256(decoded).hexdigest()
        if self.sha256 is not None and self.sha256 != digest:
            raise ValueError("sha256 与图片实际内容不一致")
        if self.filename and any(separator in self.filename for separator in ("/", "\\")):
            raise ValueError("filename 不能包含路径分隔符")

        # 计算结果写回模型，之后只需调用 metadata() 即可生成安全记录。
        object.__setattr__(self, "size_bytes", len(decoded))
        object.__setattr__(self, "sha256", digest)
        return self

    def metadata(self) -> ImageMetadata:
        """返回不含 base64 内容的安全图片元数据。"""

        return ImageMetadata(
            media_type=self.media_type,
            size_bytes=self.size_bytes or 1,
            sha256=self.sha256 or "0" * 64,
            filename=self.filename,
        )


class VisionObservation(StrictSchema):
    """视觉模型输出的受控观察结果，不等同于最终事件判断。"""

    # 描述必须是模型返回的短文本，不能直接作为真实处置规则使用。
    description: str = Field(min_length=1, max_length=2000)
    objects: list[str] = Field(default_factory=list, max_length=30)
    visible_text: list[str] = Field(default_factory=list, max_length=30)
    location_hint: str | None = Field(default=None, max_length=200)
    facility_hint: str | None = Field(default=None, max_length=200)
    hazard_signals: list[str] = Field(default_factory=list, max_length=30)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=30)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # demo 提供方会明确标记，避免把演示描述误认为真实视觉能力。
    # 默认值必须为 False：外部视觉模型没有填写此字段时，不能被错误标记为 demo。
    is_demo_observation: bool = False

    @property
    def caption(self) -> str:
        """提供 caption 别名，方便节点命名与教学代码阅读。"""

        return self.description


# 计划书使用“图片描述”这一称呼，保留一个直观的兼容别名。
ImageDescription = VisionObservation


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
    """进入 Agent 的原始文本诉求，以及可选的图片附件。"""

    # 请求唯一编号由 API 或调用方提供，后续结果使用它关联同一个案件。
    request_id: str = Field(min_length=1, max_length=100)

    # 先固定为景区服务，避免第一阶段误接入未支持的业务场景。
    scenario: ScenarioId = ScenarioId.SCENIC_SERVICE

    # 原始诉求文本；长度限制用于防止异常大的输入。
    text: str = Field(min_length=1, max_length=2000)

    # 有图片时会自动规范为 text_image，旧的纯文本调用仍保持 text。
    source_channel: InputChannel = InputChannel.TEXT

    # 未提供接收时间时，自动记录当前 UTC 时间，便于审计和排查。
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # 图片内容只在本次处理内存中使用，结果中仅保留 ImageMetadata。
    image: ImageAttachment | None = None

    @model_validator(mode="after")
    def validate_input_channel(self) -> "ServiceRequestInput":
        """确保输入渠道标记与是否带图片保持一致。"""

        if self.image is None and self.source_channel != InputChannel.TEXT:
            raise ValueError("没有图片时 source_channel 必须为 text")
        if self.image is not None and self.source_channel == InputChannel.TEXT:
            # 为兼容旧调用方，带图片但未填写渠道时自动补全标记。
            object.__setattr__(self, "source_channel", InputChannel.TEXT_IMAGE)
        elif self.image is not None and self.source_channel != InputChannel.TEXT_IMAGE:
            raise ValueError("带图片时 source_channel 必须为 text_image")
        return self


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


class MultimodalFusionStatus(str, Enum):
    """图文信息融合后的受控结论。"""

    # 图和文的可核对要点一致，可继续由其他安全规则决定是否需要人工复核。
    CONSISTENT = "consistent"
    # 图和文给出相互矛盾的地点、设施或状态线索，必须转人工。
    CONFLICT = "conflict"
    # 图片没有足够的可信线索支撑文字诉求，必须转人工。
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    # 本地 demo 没有分析像素，因而不对图文一致性作出结论。
    NOT_ASSESSED = "not_assessed"


class MultimodalConflictField(str, Enum):
    """发生图文冲突的受控字段名称，避免写入任意自由文本。"""

    LOCATION = "location"
    FACILITY = "facility"
    CONDITION = "condition"


class MultimodalFusion(StrictSchema):
    """图文融合的可追溯、安全摘要，不保存原始图片或自由文本证据。"""

    status: MultimodalFusionStatus
    # 只保存受控概念标签，例如“设施：照明设施”，不复制原始诉求或图片 Base64。
    text_concepts: list[str] = Field(default_factory=list, max_length=20)
    image_concepts: list[str] = Field(default_factory=list, max_length=20)
    conflict_fields: list[MultimodalConflictField] = Field(
        default_factory=list,
        max_length=3,
    )
    note: str = Field(min_length=1, max_length=500)
    # 只有本地确定性视觉 demo 才允许标记为 True。
    is_demo_assessment: bool = False

    @model_validator(mode="after")
    def validate_status_details(self) -> "MultimodalFusion":
        """让状态、证据摘要和冲突字段保持自洽。"""

        if self.status == MultimodalFusionStatus.CONSISTENT:
            if not self.text_concepts or not self.image_concepts:
                raise ValueError(
                    "consistent fusion requires text and image concepts"
                )
            if self.conflict_fields:
                raise ValueError("consistent fusion must not contain conflicts")
        elif self.status == MultimodalFusionStatus.CONFLICT:
            if not self.conflict_fields:
                raise ValueError("conflict fusion requires conflict_fields")
        elif self.conflict_fields:
            raise ValueError("only conflict fusion may contain conflict_fields")

        if self.status == MultimodalFusionStatus.NOT_ASSESSED:
            if not self.is_demo_assessment:
                raise ValueError("not_assessed fusion requires a demo assessment")
        elif self.is_demo_assessment:
            raise ValueError(
                "only not_assessed fusion may be marked as a demo assessment"
            )
        return self


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
    INVALID_IMAGE = "invalid_image"
    VISION_ERROR = "vision_error"
    INVALID_VISION_OUTPUT = "invalid_vision_output"
    MULTIMODAL_CONFLICT = "multimodal_conflict"
    MULTIMODAL_INSUFFICIENT_EVIDENCE = "multimodal_insufficient_evidence"


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

    # 以下字段记录可选视觉节点的调用和解析状态；没有图片时全部为 None。
    vision_call_success: bool | None = None
    vision_output_parse_success: bool | None = None
    raw_vision_output: str | None = Field(default=None, max_length=5000)
    vision_provider_name: str | None = Field(default=None, max_length=100)
    vision_model_name: str | None = Field(default=None, max_length=200)

    # 记录系统错误摘要，例如模型调用失败或规则冲突。
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_model_diagnostics(self) -> "Diagnostics":
        """保证文本模型和视觉模型的调用状态与原始输出之间没有矛盾。"""

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

        if (
            self.vision_output_parse_success is not None
            and self.vision_call_success is not True
        ):
            raise ValueError(
                "vision_output_parse_success requires a successful vision call"
            )

        if (
            self.raw_vision_output is not None
            and self.vision_output_parse_success is not False
        ):
            raise ValueError(
                "raw_vision_output is only allowed when vision parsing fails"
            )

        if (
            self.vision_output_parse_success is False
            and self.raw_vision_output is None
        ):
            raise ValueError(
                "raw_vision_output is required when vision parsing fails"
            )

        if (
            (self.vision_provider_name is not None
             or self.vision_model_name is not None)
            and self.vision_call_success is not True
        ):
            raise ValueError(
                "vision provider metadata requires a successful vision call"
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

    # 图片本身不写入结果，只保存可追溯的元数据和结构化视觉观察。
    image: ImageMetadata | None = None
    vision_observation: VisionObservation | None = None
    # 只有图片视觉输出校验成功后才会生成图文融合结论。
    multimodal_fusion: MultimodalFusion | None = None

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

        # 视觉观察和图文融合均必须与图片元数据同时存在，防止出现无法追溯的结论。
        if self.image is None and self.vision_observation is not None:
            raise ValueError("vision_observation requires image metadata")
        if self.image is None and self.multimodal_fusion is not None:
            raise ValueError("multimodal_fusion requires image metadata")
        if self.image is not None:
            if self.diagnostics.vision_call_success is not True:
                required_reasons.add(ReviewReason.VISION_ERROR)
            elif self.diagnostics.vision_output_parse_success is not True:
                required_reasons.add(ReviewReason.INVALID_VISION_OUTPUT)
            elif self.vision_observation is None:
                raise ValueError(
                    "vision_observation is required after successful vision parsing"
                )
            elif self.multimodal_fusion is None:
                raise ValueError(
                    "multimodal_fusion is required after successful vision parsing"
                )
        elif any(
            value is not None
            for value in (
                self.diagnostics.vision_call_success,
                self.diagnostics.vision_output_parse_success,
                self.diagnostics.raw_vision_output,
                self.diagnostics.vision_provider_name,
                self.diagnostics.vision_model_name,
            )
        ):
            raise ValueError(
                "vision diagnostics require image metadata"
            )

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

        # 图文冲突或视觉信息不足时，不能保留自动建议，必须交由人工确认。
        if self.multimodal_fusion is not None:
            if self.multimodal_fusion.status == MultimodalFusionStatus.CONFLICT:
                required_reasons.add(ReviewReason.MULTIMODAL_CONFLICT)
            elif (
                self.multimodal_fusion.status
                in {
                    MultimodalFusionStatus.INSUFFICIENT_EVIDENCE,
                    MultimodalFusionStatus.NOT_ASSESSED,
                }
            ):
                required_reasons.add(
                    ReviewReason.MULTIMODAL_INSUFFICIENT_EVIDENCE
                )

            if (
                self.multimodal_fusion.status
                != MultimodalFusionStatus.CONSISTENT
                and self.action_plan
            ):
                raise ValueError(
                    "action_plan must be empty when multimodal fusion is not consistent"
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
