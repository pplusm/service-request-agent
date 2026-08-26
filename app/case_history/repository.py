"""使用 SQLite 保存本地演示案件结果，并按人工复核状态查询。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.case_history.models import CaseHistoryRecord
from app.schemas.models import ServiceCaseResult


_MAX_RETURNED_RECORDS = 100
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
        """在首次启动时创建表和索引；已有数据库不会清除历史记录。"""

        try:
            with self._connection() as connection:
                with connection:
                    connection.executescript(_CREATE_TABLE_STATEMENT)
        except sqlite3.Error as error:
            raise CaseHistoryStorageError(
                f"failed to initialize local case history: {error}"
            ) from error

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
