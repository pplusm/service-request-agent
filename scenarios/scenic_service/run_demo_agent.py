"""在本地终端演示完整的景区服务诉求 Agent 工作流。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Final

from app.agent.triage_graph import TriageAgent
from app.llm.mock_provider import MockLLMProvider
from app.rag.knowledge_store import ChromaKnowledgeStore
from app.schemas.models import KnowledgeReference, ServiceCaseResult, ServiceRequestInput


# 从脚本位置回到项目根目录，使脚本可以在项目根目录以外被执行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "data" / "scenic_service" / "knowledge"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_data"

# 以下词语只用于本地教学演示的确定性文本匹配，不是实际的语言模型能力。
_DEMO_LOCATION_PATTERN: Final = re.compile(
    r"(?P<location>(?:东|西|南|北)门(?:附近)?|游客中心(?:附近)?|"
    r"停车场(?:附近)?|入口(?:附近)?|出口(?:附近)?|服务台(?:附近)?|"
    r"观景台(?:附近)?|广场(?:附近)?)"
)
_HIGH_RISK_KEYWORDS: Final = (
    "受伤",
    "昏倒",
    "晕倒",
    "摔倒",
    "流血",
    "火灾",
    "起火",
    "烟雾",
    "危险",
    "急救",
)


def run_agent(
    store: ChromaKnowledgeStore,
    request_id: str,
    text: str,
    *,
    simulate_invalid_output: bool = False,
) -> ServiceCaseResult:
    """执行一次完整流程，并始终返回经过 Pydantic 校验的案件结果。"""

    service_request = ServiceRequestInput(request_id=request_id, text=text)

    # 先用与 Agent 相同的知识库预览本次资料，以便模拟输出引用实际资料。
    # Agent 在工作流中仍会自行再次检索；这一步不替代或绕过 RAG 节点。
    references = store.search(service_request.text, limit=1)
    provider = MockLLMProvider(
        response_content=_build_mock_response(
            service_request,
            references,
            simulate_invalid_output=simulate_invalid_output,
        )
    )
    agent = TriageAgent(knowledge_store=store, model_provider=provider)
    return agent.run(service_request)


def _build_mock_response(
    service_request: ServiceRequestInput,
    references: list[KnowledgeReference],
    *,
    simulate_invalid_output: bool,
) -> str:
    """按可控的演示规则生成模拟模型 JSON，绝不连接任何外部服务。"""

    if simulate_invalid_output:
        # 只有用户显式开启该开关时，才模拟模型输出不合格的情形。
        return "{}"

    # 知识未命中时 LangGraph 会直接跳过模型调用，因此这里的文本不会被使用。
    if not references:
        return "{}"

    issue = _identify_demo_issue(service_request.text)
    if issue is None:
        # 虽然向量检索到相近资料，但资料未明确支持该诉求时不把它当作依据。
        return _build_unsupported_issue_response(service_request)

    facility_name, evidence = issue
    location = _extract_demo_location(service_request.text)
    if _contains_high_risk_signal(service_request.text):
        return _build_high_risk_response(
            service_request=service_request,
            reference=references[0],
            location=location,
            facility_name=facility_name,
            evidence=evidence,
        )

    return _build_facility_fault_response(
        service_request=service_request,
        reference=references[0],
        location=location,
        facility_name=facility_name,
        evidence=evidence,
    )


def _identify_demo_issue(text: str) -> tuple[str, str] | None:
    """识别演示资料明确列出的三类设施问题，并保留输入中的原始证据。"""

    normalized_text = text.strip()
    if ("卫生间" in normalized_text or "洗手间" in normalized_text or "厕所" in normalized_text) and (
        "没水" in normalized_text
        or "无水" in normalized_text
        or "停水" in normalized_text
    ):
        return "卫生间", normalized_text

    if "指示牌" in normalized_text and (
        "损坏" in normalized_text
        or "破损" in normalized_text
        or "坏了" in normalized_text
    ):
        return "指示牌", normalized_text

    if ("照明" in normalized_text or "路灯" in normalized_text) and (
        "故障" in normalized_text
        or "损坏" in normalized_text
        or "不亮" in normalized_text
        or "坏了" in normalized_text
    ):
        return "照明设施", normalized_text

    return None


def _extract_demo_location(text: str) -> str | None:
    """从有限的演示地点词中提取位置；未找到时必须转人工确认。"""

    match = _DEMO_LOCATION_PATTERN.search(text)
    if match is None:
        return None
    return match.group("location")


def _contains_high_risk_signal(text: str) -> bool:
    """识别演示用高风险提示词；该判断不构成真实安全评估。"""

    return any(keyword in text for keyword in _HIGH_RISK_KEYWORDS)


def _build_facility_fault_response(
    *,
    service_request: ServiceRequestInput,
    reference: KnowledgeReference,
    location: str | None,
    facility_name: str,
    evidence: str,
) -> str:
    """构造普通设施问题的结果；地点缺失时保留分类但要求人工复核。"""

    missing_fields = [] if location is not None else ["location"]
    requires_human_review = bool(missing_fields)
    action_plan: list[dict[str, object]] = []
    if not requires_human_review:
        # 该建议明确标记为演示，不能理解为真实景区调度指令。
        action_plan.append(
            {
                "step": 1,
                "suggested_action": "创建演示性设施维护跟进建议。",
                "knowledge_source_ids": [reference.source_id],
                "is_demo_action": True,
            }
        )

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": facility_name,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": "facility_fault",
            # 这是固定演示分数，不是系统准确率或真实模型指标。
            "confidence": 0.9,
            "evidence": [evidence],
        },
        risk={
            "level": "low",
            "risk_factors": [],
            "summary": "仅用于项目演示的低风险设施故障判断。",
        },
        knowledge_references=[reference.model_dump(mode="json")],
        action_plan=action_plan,
        review={
            "requires_human_review": requires_human_review,
            "reasons": ["missing_fields"] if requires_human_review else [],
            "review_note": "地点信息缺失，需人工补充确认。"
            if requires_human_review
            else "",
        },
        diagnostics={
            "knowledge_hit": True,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    )


def _build_high_risk_response(
    *,
    service_request: ServiceRequestInput,
    reference: KnowledgeReference,
    location: str | None,
    facility_name: str,
    evidence: str,
) -> str:
    """高风险提示出现时构造可解析结果，但绝不输出自动处置建议。"""

    missing_fields = [] if location is not None else ["location"]
    review_reasons = ["high_risk"]
    if missing_fields:
        review_reasons.append("missing_fields")

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": facility_name,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": "facility_fault",
            "confidence": 0.9,
            "evidence": [evidence],
        },
        risk={
            "level": "high",
            "risk_factors": ["文本含有可能的安全或健康风险描述。"],
            "summary": "演示性高风险提示，必须由人工确认。",
        },
        knowledge_references=[reference.model_dump(mode="json")],
        action_plan=[],
        review={
            "requires_human_review": True,
            "reasons": review_reasons,
            "review_note": "检测到可能的高风险描述，已转人工复核。",
        },
        diagnostics={
            "knowledge_hit": True,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    )


def _build_unsupported_issue_response(
    service_request: ServiceRequestInput,
) -> str:
    """相近资料不足以支持诉求时，返回可校验的保守人工复核结果。"""

    location = _extract_demo_location(service_request.text)
    missing_fields = ["facility_name"]
    if location is None:
        missing_fields.insert(0, "location")

    return _serialize_result_payload(
        service_request=service_request,
        entities={
            "location": location,
            "facility_name": None,
            "visitor_condition": None,
            "estimated_affected_count": None,
            "event_time_description": None,
            "missing_fields": missing_fields,
        },
        classification={
            "event_type": "other_unknown",
            "confidence": 0.0,
            "evidence": [],
        },
        risk={
            "level": "unassessed",
            "risk_factors": [],
            "summary": "当前无法安全评估风险，需人工复核。",
        },
        # 相近资料不能作为本案依据，因此不保留它的引用或处置建议。
        knowledge_references=[],
        action_plan=[],
        review={
            "requires_human_review": True,
            "reasons": [
                "unassessed_risk",
                "missing_fields",
                "knowledge_not_found",
                "low_confidence",
            ],
            "review_note": "演示资料未明确支持该诉求，已转人工复核。",
        },
        diagnostics={
            "knowledge_hit": False,
            "model_call_success": True,
            "model_output_parse_success": True,
            "raw_model_output": None,
            "errors": [],
        },
    )


def _serialize_result_payload(
    *,
    service_request: ServiceRequestInput,
    entities: dict[str, object],
    classification: dict[str, object],
    risk: dict[str, object],
    knowledge_references: list[dict[str, object]],
    action_plan: list[dict[str, object]],
    review: dict[str, object],
    diagnostics: dict[str, object],
) -> str:
    """将模拟结果统一序列化为 JSON，随后仍由 Agent 的 Pydantic 解析器校验。"""

    payload = {
        "request_id": service_request.request_id,
        "scenario": service_request.scenario.value,
        "entities": entities,
        "classification": classification,
        "risk": risk,
        "knowledge_references": knowledge_references,
        "action_plan": action_plan,
        "review": review,
        "diagnostics": diagnostics,
    }
    return json.dumps(payload, ensure_ascii=False)


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
