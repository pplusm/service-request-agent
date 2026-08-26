"""在本地终端演示完整的景区服务诉求 Agent 工作流。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.agent.triage_graph import TriageAgent
from app.llm.demo_provider import DemoLLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import ServiceCaseResult, ServiceRequestInput


# 从脚本位置回到项目根目录，使脚本可以在项目根目录以外被执行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "data" / "scenic_service" / "knowledge"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"


def run_agent(
    store: ChromaKnowledgeStore,
    request_id: str,
    text: str,
    *,
    simulate_invalid_output: bool = False,
) -> ServiceCaseResult:
    """执行一次完整流程，并始终返回经过 Pydantic 校验的案件结果。"""

    service_request = ServiceRequestInput(request_id=request_id, text=text)
    provider = DemoLLMProvider(
        simulate_invalid_output=simulate_invalid_output,
    )
    agent = TriageAgent(knowledge_store=store, model_provider=provider)
    return agent.run(service_request)


def main() -> None:
    """支持单次命令行调用，也支持在终端中连续输入诉求。"""

    if hasattr(sys.stdout, "reconfigure"):
        # 显式使用 UTF-8，避免 Windows 终端中中文 JSON 出现乱码。
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="运行本地景区服务 Agent 演示。")
    parser.add_argument("text", nargs="?", help="可选的景区服务文本诉求。")
    parser.add_argument(
        "--simulate-invalid-output",
        action="store_true",
        help="故意模拟模型输出不合格，用于演示人工复核兜底。",
    )
    arguments = parser.parse_args()

    store = ChromaKnowledgeStore(persist_directory=CHROMA_DIRECTORY)
    store.index_directory(KNOWLEDGE_DIRECTORY)

    if arguments.text:
        result = run_agent(
            store,
            request_id="demo_agent_001",
            text=arguments.text,
            simulate_invalid_output=arguments.simulate_invalid_output,
        )
        print(result.model_dump_json(indent=2))
        return

    print("请输入景区服务文本；输入 q 后按 Enter 退出。")
    sequence = 1
    while True:
        text = input("> ").strip()
        if text.lower() == "q":
            return
        if not text:
            print("请输入非空文本。")
            continue

        result = run_agent(
            store,
            request_id=f"demo_agent_{sequence:03d}",
            text=text,
            simulate_invalid_output=arguments.simulate_invalid_output,
        )
        print(result.model_dump_json(indent=2))
        sequence += 1


if __name__ == "__main__":
    main()
