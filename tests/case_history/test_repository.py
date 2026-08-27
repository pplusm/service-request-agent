"""验证本地案件历史只保存合规结果，并正确筛选人工复核队列。"""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from app.agent.result_parser import build_input_validation_result
from app.case_history.models import CaseHistoryRecord
from app.case_history.repository import LocalCaseHistoryRepository
from app.schemas.models import (
    MultimodalFusionStatus,
    ReviewReason,
    ServiceCaseResult,
)


def build_normal_result() -> ServiceCaseResult:
    """构造一条不需要人工复核的最小合规演示结果。"""

    return ServiceCaseResult.model_validate(
        {
            "request_id": "history_normal_001",
            "scenario": "scenic_service",
            "entities": {
                "location": "西门",
                "facility_name": "照明设施",
                "visitor_condition": None,
                "estimated_affected_count": None,
                "event_time_description": None,
                "missing_fields": [],
            },
            "classification": {
                "event_type": "facility_fault",
                "confidence": 0.9,
                "evidence": ["演示性设施故障证据"],
            },
            "risk": {
                "level": "low",
                "risk_factors": [],
                "summary": "仅用于本地仓库测试的演示结果。",
            },
            "knowledge_references": [
                {
                    "source_id": "demo_facility_001",
                    "source_title": "Demo facility guidance",
                    "source_path": "data/scenic_service/knowledge/demo_facility.md",
                    "excerpt": "仅用于本地仓库测试的演示资料。",
                    "relevance_score": 0.9,
                    "is_demo_source": True,
                }
            ],
            "action_plan": [
                {
                    "step": 1,
                    "suggested_action": "创建演示性设施维护跟进建议。",
                    "knowledge_source_ids": ["demo_facility_001"],
                    "is_demo_action": True,
                }
            ],
            "review": {
                "requires_human_review": False,
                "reasons": [],
                "review_note": "",
            },
            "diagnostics": {
                "knowledge_hit": True,
                "model_call_success": True,
                "model_output_parse_success": True,
                "raw_model_output": None,
                "errors": [],
            },
        }
    )


def test_repository_saves_history_and_filters_human_review_queue(
    tmp_path: Path,
) -> None:
    """普通案件只进入历史，安全兜底案件还必须进入待人工复核列表。"""

    repository = LocalCaseHistoryRepository(tmp_path / "case_history.sqlite3")
    normal_record = repository.save(build_normal_result())
    review_record = repository.save(
        build_input_validation_result(
            request_id="history_review_001",
            missing_fields=["text"],
            validation_errors=["missing or empty field: text"],
        )
    )

    history = repository.list_recent()
    review_queue = repository.list_pending_human_review()

    assert {record.record_id for record in history} == {
        normal_record.record_id,
        review_record.record_id,
    }
    assert [record.record_id for record in review_queue] == [
        review_record.record_id
    ]
    assert review_queue[0].requires_human_review is True
    assert CaseHistoryRecord.model_validate_json(
        review_record.model_dump_json()
    ) == review_record


def test_repository_migrates_legacy_vision_record_to_human_review(
    tmp_path: Path,
) -> None:
    """旧版图片记录补齐融合字段后，必须持久化为人工复核案件。"""

    database_path = tmp_path / "case_history.sqlite3"
    repository = LocalCaseHistoryRepository(database_path)
    legacy_payload = build_normal_result().model_dump(mode="json")
    legacy_payload.update(
        {
            "request_id": "history_legacy_image_001",
            "image": {
                "media_type": "image/png",
                "size_bytes": 1,
                "sha256": "0" * 64,
                "filename": "legacy-demo.png",
            },
            "vision_observation": {
                "description": "本地演示视觉输出，未分析真实图片像素。",
                "objects": [],
                "visible_text": [],
                "location_hint": None,
                "facility_hint": None,
                "hazard_signals": [],
                "uncertainty_notes": [],
                "confidence": 0.0,
                "is_demo_observation": True,
            },
        }
    )
    legacy_payload["diagnostics"].update(
        {
            "vision_call_success": True,
            "vision_output_parse_success": True,
            "raw_vision_output": None,
            "vision_provider_name": "local_demo_vision",
            "vision_model_name": "deterministic-demo-vision",
        }
    )

    # 模拟“图文融合字段发布前”已写入的旧 JSON，故意保持旧的自动处置状态。
    legacy_record_id = uuid4()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO case_history (
                record_id, request_id, recorded_at, requires_human_review, result_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(legacy_record_id),
                legacy_payload["request_id"],
                "2026-08-27T00:00:00+00:00",
                0,
                json.dumps(legacy_payload, ensure_ascii=False),
            ),
        )

    # 再次初始化仓库会运行兼容迁移；它不得删除记录或把它当成自动处置案件。
    migrated_repository = LocalCaseHistoryRepository(database_path)
    history = migrated_repository.list_recent()
    review_queue = migrated_repository.list_pending_human_review()

    assert len(history) == 1
    migrated_record = history[0]
    assert migrated_record.record_id == legacy_record_id
    assert migrated_record.requires_human_review is True
    assert migrated_record.result.action_plan == []
    assert migrated_record.result.multimodal_fusion is not None
    assert (
        migrated_record.result.multimodal_fusion.status
        == MultimodalFusionStatus.NOT_ASSESSED
    )
    assert migrated_record.result.multimodal_fusion.is_demo_assessment is True
    assert (
        ReviewReason.MULTIMODAL_INSUFFICIENT_EVIDENCE
        in migrated_record.result.review.reasons
    )
    assert [record.record_id for record in review_queue] == [legacy_record_id]

    # 迁移不仅影响内存中的展示结果，也同步更新 SQLite 的人工复核索引和 JSON。
    with sqlite3.connect(database_path) as connection:
        stored_row = connection.execute(
            """
            SELECT requires_human_review, result_json
            FROM case_history
            WHERE record_id = ?
            """,
            (str(legacy_record_id),),
        ).fetchone()
    assert stored_row is not None
    assert stored_row[0] == 1
    stored_payload = json.loads(stored_row[1])
    assert stored_payload["multimodal_fusion"]["status"] == "not_assessed"
