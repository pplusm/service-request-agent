"""使用 SQLite 保存本地演示案件结果，并按人工复核状态查询。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.case_history.models import CaseHistoryRecord
from app.schemas.models import ServiceCaseResult


_MAX_RETURNED_RECORDS = 100
_LEGACY_FUSION_REVIEW_NOTE = (
    "历史记录创建于图文融合字段上线前，无法补充核对，需人工复核。"
)
_CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS case_history (
    record_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    requires_human_review INTEGER NOT NULL CHECK (requires_human_review IN (0, 1)),
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_case_history_recorded_at
ON case_history (recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_case_history_pending_review
ON case_history (requires_human_review, recorded_at DESC);
"""


class CaseHistoryStorageError(RuntimeError):
    """本地 SQLite 无法安全读写案件记录时抛出的异常。"""


class LocalCaseHistoryRepository:
    """本地演示案件历史仓库，每次操作都会重新打开 SQLite 连接。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def save(self, result: ServiceCaseResult) -> CaseHistoryRecord:
        """再次校验 Agent 结果后，将其保存为一条新的本地历史记录。"""

        # 先经历 JSON 往返校验，避免把未符合核心契约的对象写入本地数据库。
        result_json = result.model_dump_json()
        validated_result = ServiceCaseResult.model_validate_json(result_json)
        record = CaseHistoryRecord(
            record_id=uuid4(),
            recorded_at=datetime.now(timezone.utc),
            requires_human_review=validated_result.review.requires_human_review,
            result=validated_result,
        )

        try:
            with self._connection() as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO case_history (
                            record_id,
                            request_id,
                            recorded_at,
                            requires_human_review,
                            result_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            str(record.record_id),
                            record.result.request_id,
                            record.recorded_at.isoformat(),
                            int(record.requires_human_review),
                            result_json,
                        ),
                    )
        except sqlite3.Error as error:
            raise CaseHistoryStorageError(
                f"failed to save local case history: {error}"
            ) from error
        return record

    def list_recent(self) -> list[CaseHistoryRecord]:
        """返回最近保存的 100 条本地演示案件记录。"""

        return self._list_records(
            """
            SELECT record_id, recorded_at, requires_human_review, result_json
            FROM case_history
            ORDER BY recorded_at DESC, record_id DESC
            LIMIT ?
            """
        )

    def list_pending_human_review(self) -> list[CaseHistoryRecord]:
        """返回最近 100 条必须人工复核的本地案件。"""

        return self._list_records(
            """
            SELECT record_id, recorded_at, requires_human_review, result_json
            FROM case_history
            WHERE requires_human_review = 1
            ORDER BY recorded_at DESC, record_id DESC
            LIMIT ?
            """
        )

    def _initialize_database(self) -> None:
        """创建表和索引，并保守迁移已知的旧版图片历史记录。"""

        try:
            with self._connection() as connection:
                with connection:
                    connection.executescript(_CREATE_TABLE_STATEMENT)
                    self._migrate_legacy_multimodal_records(connection)
        except sqlite3.Error as error:
            raise CaseHistoryStorageError(
                f"failed to initialize local case history: {error}"
            ) from error

    @staticmethod
    def _migrate_legacy_multimodal_records(
        connection: sqlite3.Connection,
    ) -> None:
        """补齐图文融合字段上线前的已知图片记录，并强制转人工复核。

        只处理视觉调用和解析都成功、且恰好缺少 ``multimodal_fusion`` 的
        旧结构。其他损坏或不明格式的 JSON 保持原样，仍由后续严格校验拒绝，
        避免把未知数据伪装成可信案件。
        """

        rows = connection.execute(
            "SELECT record_id, result_json FROM case_history"
        ).fetchall()
        updates: list[tuple[int, str, str]] = []

        for row in rows:
            migrated_result_json = _build_legacy_multimodal_result_json(
                str(row["result_json"])
            )
            if migrated_result_json is None:
                continue

            try:
                # 写回前仍需完整通过当前 Pydantic 契约，不能依赖迁移条件本身。
                validated_result = ServiceCaseResult.model_validate_json(
                    migrated_result_json
                )
            except ValidationError:
                # 旧记录还有其他不安全问题时不擅自修补，保留给严格读取逻辑处理。
                continue

            updates.append(
                (
                    int(validated_result.review.requires_human_review),
                    validated_result.model_dump_json(),
                    str(row["record_id"]),
                )
            )

        if updates:
            connection.executemany(
                """
                UPDATE case_history
                SET requires_human_review = ?, result_json = ?
                WHERE record_id = ?
                """,
                updates,
            )

    def _list_records(self, query: str) -> list[CaseHistoryRecord]:
        """执行只读查询，并把每条 JSON 重新解析为严格 Pydantic 模型。"""

        try:
            with self._connection() as connection:
                rows = connection.execute(query, (_MAX_RETURNED_RECORDS,)).fetchall()
            return [self._row_to_record(row) for row in rows]
        except (sqlite3.Error, ValidationError, ValueError) as error:
            raise CaseHistoryStorageError(
                f"failed to read local case history: {error}"
            ) from error

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CaseHistoryRecord:
        """拒绝被篡改或损坏的数据库行，不将未经校验的 JSON 暴露给页面。"""

        return CaseHistoryRecord(
            record_id=UUID(str(row["record_id"])),
            recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
            requires_human_review=bool(row["requires_human_review"]),
            result=ServiceCaseResult.model_validate_json(str(row["result_json"])),
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """为每次读写提供独立连接，避免 FastAPI 请求间共享线程状态。"""

        connection = sqlite3.connect(
            self._database_path,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _build_legacy_multimodal_result_json(result_json: str) -> str | None:
    """把唯一可识别的旧版视觉记录转换为保守的当前 JSON 结构。

    本函数只填充当时尚不存在的图文融合结论及其必需的人工复核标记；
    不会推测图片内容，也不会修复其他字段错误。
    """

    try:
        payload: Any = json.loads(result_json)
    except json.JSONDecodeError:
        return None

    # 旧版序列化可能完全没有该键，也可能按当时的默认值写入 null。
    # 只有已有非空融合结论的当前记录才不应被迁移。
    if (
        not isinstance(payload, dict)
        or payload.get("multimodal_fusion") is not None
    ):
        return None

    image = payload.get("image")
    observation = payload.get("vision_observation")
    diagnostics = payload.get("diagnostics")
    review = payload.get("review")
    if not (
        isinstance(image, dict)
        and isinstance(observation, dict)
        and isinstance(diagnostics, dict)
        and isinstance(review, dict)
    ):
        return None

    # 只有当旧记录已明确表示视觉调用和输出解析成功时，才属于已知兼容场景。
    if (
        diagnostics.get("vision_call_success") is not True
        or diagnostics.get("vision_output_parse_success") is not True
    ):
        return None

    reasons = review.get("reasons")
    review_note = review.get("review_note")
    if (
        not isinstance(reasons, list)
        or not all(isinstance(reason, str) for reason in reasons)
        or not isinstance(review_note, str)
    ):
        return None

    # 本地 demo 没有识别真实像素，只能标为“未评估”；其他旧视觉结果也一律保守处理。
    is_demo_observation = observation.get("is_demo_observation") is True
    if is_demo_observation:
        fusion_status = "not_assessed"
        fusion_note = "历史本地演示视觉模型未分析图片像素，无法核对图文信息。"
    else:
        fusion_status = "insufficient_evidence"
        fusion_note = "历史视觉记录缺少图文融合依据，无法确认图文一致性。"

    payload["multimodal_fusion"] = {
        "status": fusion_status,
        "text_concepts": [],
        "image_concepts": [],
        "conflict_fields": [],
        "note": fusion_note,
        "is_demo_assessment": is_demo_observation,
    }
    # 非一致融合结果不能沿用旧的自动建议，必须进入人工复核队列。
    payload["action_plan"] = []
    review["requires_human_review"] = True
    if "multimodal_insufficient_evidence" not in reasons:
        review["reasons"] = [
            *reasons,
            "multimodal_insufficient_evidence",
        ]
    combined_note = " ".join(
        note for note in (review_note.strip(), _LEGACY_FUSION_REVIEW_NOTE) if note
    )
    review["review_note"] = combined_note[:500]
    return json.dumps(payload, ensure_ascii=False)
