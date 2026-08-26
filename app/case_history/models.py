"""定义本地案件历史和待人工复核列表的 Pydantic 输出模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.models import ServiceCaseResult, StrictSchema


class CaseHistoryRecord(StrictSchema):
    """一条持久化的本地案件结果，不单独保存原始请求体。"""

    # 数据库记录 ID 与调用方提供的 request_id 分开，允许同一案件编号重复演示。
    record_id: UUID
    recorded_at: datetime
    requires_human_review: bool
    result: ServiceCaseResult

    @model_validator(mode="after")
    def validate_review_flag(self) -> "CaseHistoryRecord":
        """数据库索引字段必须与核心案件结果的复核标记保持一致。"""

        if self.requires_human_review != self.result.review.requires_human_review:
            raise ValueError(
                "requires_human_review must match result.review.requires_human_review"
            )
        return self


class CaseHistoryResponse(StrictSchema):
    """案件历史接口的可校验 JSON 响应。"""

    # 第一阶段页面只展示最近 100 条本地演示记录，避免列表无限增长。
    records: list[CaseHistoryRecord] = Field(default_factory=list, max_length=100)
    storage_error: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_storage_error(self) -> "CaseHistoryResponse":
        """读取失败时不混合展示可能不完整的历史数据。"""

        if self.storage_error is not None and self.records:
            raise ValueError("records must be empty when storage_error is present")
        return self


class HumanReviewQueueResponse(StrictSchema):
    """待人工复核列表接口的可校验 JSON 响应。"""

    records: list[CaseHistoryRecord] = Field(default_factory=list, max_length=100)
    storage_error: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_pending_review_records(self) -> "HumanReviewQueueResponse":
        """队列中只允许出现由安全规则要求人工复核的记录。"""

        if self.storage_error is not None and self.records:
            raise ValueError("records must be empty when storage_error is present")
        if any(not record.requires_human_review for record in self.records):
            raise ValueError(
                "human review queue records must require human review"
            )
        return self
