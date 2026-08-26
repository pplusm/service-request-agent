"""验证 LangGraph 演示 Agent 的端到端安全行为。"""

import json
from pathlib import Path

from app.agent.triage_graph import TriageAgent
from app.llm.mock_provider import MockLLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import (
    KnowledgeReference,
    ReviewReason,
    ServiceRequestInput,
)


def build_store(tmp_path: Path) -> ChromaKnowledgeStore:
    """创建写入临时目录的本地演示知识库。"""

    project_root = Path(__file__).resolve().parents[2]
    knowledge_directory = project_root / "data" / "scenic_service" / "knowledge"
    store = ChromaKnowledgeStore(persist_directory=tmp_path / "chroma")
    store.index_directory(knowledge_directory)
    return store


def build_valid_model_output(reference: KnowledgeReference) -> str:
    """构造一份带真实检索引用的模拟模型 JSON。"""

    payload = {
        "request_id": "graph_demo_001",
        "scenario": "scenic_service",
        "entities": {
            "location": "东门附近",
            "facility_name": "卫生间",
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": [],
        },
        "classification": {
            "event_type": "facility_fault",
            "confidence": 0.9,
            "evidence": ["卫生间没水"],
        },
        "risk": {
            "level": "low",
            "risk_factors": [],
            "summary": "演示性低风险设施故障。",
        },
        "knowledge_references": [reference.model_dump(mode="json")],
        "action_plan": [
            {
                "step": 1,
                "suggested_action": "创建演示性设施维护跟进建议。",
                "knowledge_source_ids": [reference.source_id],
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
    return json.dumps(payload, ensure_ascii=False)


def test_triage_agent_runs_retrieval_mock_model_and_parser(tmp_path: Path) -> None:
    """知识命中时，LangGraph 应依次执行检索、模拟模型和安全解析。"""

    store = build_store(tmp_path)
    reference = store.search("东门附近卫生间没水", limit=1)[0]
    provider = MockLLMProvider(build_valid_model_output(reference))
    agent = TriageAgent(knowledge_store=store, model_provider=provider)

    result = agent.run(
        ServiceRequestInput(
            request_id="graph_demo_001",
            text="东门附近卫生间没水。",
        )
    )

    assert result.review.requires_human_review is False
    assert result.classification.event_type.value == "facility_fault"
    assert result.knowledge_references[0].source_id == "demo_facility_001"
    assert len(provider.requests) == 1
    assert provider.requests[0].schema_name == "service_case_result"


def test_triage_agent_skips_model_when_knowledge_is_not_found(tmp_path: Path) -> None:
    """无关诉求必须直接人工复核，且不产生一次模拟模型调用。"""

    store = build_store(tmp_path)
    provider = MockLLMProvider()
    agent = TriageAgent(knowledge_store=store, model_provider=provider)

    result = agent.run(
        ServiceRequestInput(
            request_id="graph_demo_002",
            text="航班改签咨询。",
        )
    )

    assert result.review.requires_human_review is True
    assert ReviewReason.KNOWLEDGE_NOT_FOUND in result.review.reasons
    assert result.diagnostics.model_call_success is None
    assert provider.requests == []


def test_triage_agent_turns_invalid_mock_output_into_human_review(
    tmp_path: Path,
) -> None:
    """模型返回非法 JSON 时，图仍必须输出可校验的人工复核结果。"""

    store = build_store(tmp_path)
    provider = MockLLMProvider(response_content="这不是 JSON")
    agent = TriageAgent(knowledge_store=store, model_provider=provider)

    result = agent.run(
        ServiceRequestInput(
            request_id="graph_demo_003",
            text="东门附近卫生间没水。",
        )
    )

    assert result.review.requires_human_review is True
    assert ReviewReason.INVALID_MODEL_OUTPUT in result.review.reasons
    assert result.diagnostics.model_call_success is True
    assert result.diagnostics.model_output_parse_success is False


def test_triage_agent_turns_provider_error_into_human_review(tmp_path: Path) -> None:
    """模型提供方报错时，图必须返回 LLM_ERROR 人工复核结果。"""

    store = build_store(tmp_path)
    provider = MockLLMProvider(error_message="演示提供方不可用")
    agent = TriageAgent(knowledge_store=store, model_provider=provider)

    result = agent.run(
        ServiceRequestInput(
            request_id="graph_demo_004",
            text="东门附近卫生间没水。",
        )
    )

    assert result.review.requires_human_review is True
    assert ReviewReason.LLM_ERROR in result.review.reasons
    assert result.diagnostics.model_call_success is False
    assert result.diagnostics.model_output_parse_success is None
