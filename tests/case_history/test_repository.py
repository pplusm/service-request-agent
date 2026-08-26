"""验证本地案件历史只保存合规结果，并正确筛选人工复核队列。"""

from pathlib import Path

from app.agent.result_parser import build_input_validation_result
from app.case_history.models import CaseHistoryRecord
from app.case_history.repository import LocalCaseHistoryRepository
from app.schemas.models import ServiceCaseResult


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
