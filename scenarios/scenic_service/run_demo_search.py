"""在本地终端演示 ChromaDB 知识检索，不调用任何外部模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import KnowledgeSearchResult


# 从脚本位置回到项目根目录，避免用户必须手动填写绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "data" / "scenic_service" / "knowledge"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"


def run_search(store: ChromaKnowledgeStore, query: str) -> KnowledgeSearchResult:
    """执行一次检索，并将结果包装为可校验的 Pydantic JSON。"""

    references = store.search(query)
    return KnowledgeSearchResult(
        query=query,
        knowledge_hit=bool(references),
        knowledge_references=references,
    )


def main() -> None:
    """支持单次命令行查询，也支持连续的交互式查询。"""

    if hasattr(sys.stdout, "reconfigure"):
        # VS Code 终端通常使用 UTF-8；显式设置可减少中文 JSON 显示异常。
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the local demo knowledge search.")
    parser.add_argument("query", nargs="?", help="Optional scenic-service request text.")
    arguments = parser.parse_args()

    store = ChromaKnowledgeStore(persist_directory=CHROMA_DIRECTORY)
    indexed_source_ids = store.index_directory(KNOWLEDGE_DIRECTORY)
    print(f"Indexed demo sources: {', '.join(indexed_source_ids)}")

    if arguments.query:
        print(run_search(store, arguments.query).model_dump_json(indent=2))
        return

    print("Enter a request text. Type q and press Enter to exit.")
    while True:
        query = input("> ").strip()
        if query.lower() == "q":
            return
        if not query:
            print("Please enter non-empty text.")
            continue
        print(run_search(store, query).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
