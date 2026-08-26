from pathlib import Path

import pytest

from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import KnowledgeSearchResult


def test_demo_knowledge_documents_are_indexed_and_cited(tmp_path: Path) -> None:
    """十篇演示知识文件应能入库，并在设施查询中保留可追溯来源。"""

    project_root = Path(__file__).resolve().parents[2]
    knowledge_directory = project_root / "data" / "scenic_service" / "knowledge"
    persist_directory = tmp_path / "chroma"
    store = ChromaKnowledgeStore(persist_directory=persist_directory)

    indexed_source_ids = store.index_directory(knowledge_directory)

    # 用同一持久化目录重新创建对象，模拟服务重启后继续检索的情况。
    reopened_store = ChromaKnowledgeStore(persist_directory=persist_directory)
    references = reopened_store.search("卫生间没水", limit=1)

    assert set(indexed_source_ids) == {
        "demo_crowd_001",
        "demo_crowd_002",
        "demo_facility_001",
        "demo_guest_service_001",
        "demo_health_001",
        "demo_health_002",
        "demo_hygiene_001",
        "demo_hygiene_002",
        "demo_safety_001",
        "demo_safety_002",
    }
    assert reopened_store.count() == 10
    assert len(references) == 1
    assert references[0].source_id == "demo_facility_001"
    assert references[0].source_path == (
        "data/scenic_service/knowledge/demo_facility.md"
    )
    assert references[0].is_demo_source is True
    assert 0.0 <= references[0].relevance_score <= 1.0

    # 演示终端输出也使用 Pydantic 模型，避免拼接未校验的 JSON 字典。
    search_result = KnowledgeSearchResult(
        query="卫生间没水",
        knowledge_hit=True,
        knowledge_references=references,
    )
    assert '"knowledge_hit":true' in search_result.model_dump_json()


@pytest.mark.parametrize(
    ("query", "expected_source_id"),
    [
        ("客流拥挤", "demo_crowd_001"),
        ("检票排队", "demo_crowd_002"),
        ("地面污渍", "demo_hygiene_001"),
        ("垃圾桶已满", "demo_hygiene_002"),
        ("身体不适", "demo_health_001"),
        ("医疗咨询", "demo_health_002"),
        ("儿童走失", "demo_safety_001"),
        ("护栏异常", "demo_safety_002"),
        ("无障碍通道", "demo_guest_service_001"),
    ],
)
def test_each_new_demo_topic_has_a_traceable_source(
    tmp_path: Path,
    query: str,
    expected_source_id: str,
) -> None:
    """每个新增主题的明确演示关键词都应返回对应的资料来源。"""

    project_root = Path(__file__).resolve().parents[2]
    knowledge_directory = project_root / "data" / "scenic_service" / "knowledge"
    store = ChromaKnowledgeStore(persist_directory=tmp_path / "chroma")
    store.index_directory(knowledge_directory)

    references = store.search(query, limit=3)

    assert expected_source_id in [reference.source_id for reference in references]


def test_unrelated_request_returns_no_knowledge_reference(tmp_path: Path) -> None:
    """无关诉求不能被“最相近”的演示资料误判为知识命中。"""

    project_root = Path(__file__).resolve().parents[2]
    knowledge_directory = project_root / "data" / "scenic_service" / "knowledge"
    store = ChromaKnowledgeStore(persist_directory=tmp_path / "chroma")
    store.index_directory(knowledge_directory)

    references = store.search("航班改签咨询", limit=1)
    search_result = KnowledgeSearchResult(
        query="航班改签咨询",
        knowledge_hit=False,
        knowledge_references=references,
    )

    assert references == []
    assert search_result.knowledge_hit is False
