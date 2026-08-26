"""验证景区服务 YAML 配置及其与演示模型的连接。"""

import json
from pathlib import Path

import pytest

from app.llm.demo_provider import DemoLLMProvider
from app.llm.provider import ChatMessage, StructuredGenerationRequest
from app.rules.scenic_service_config import (
    ScenarioConfigurationError,
    load_scenic_service_configuration,
)
from app.schemas.models import KnowledgeReference, RiskLevel, ServiceRequestInput


def build_provider_request(
    text: str,
    reference: KnowledgeReference,
) -> StructuredGenerationRequest:
    """构造与 Agent 传给模型一致的最小结构化请求。"""

    service_request = ServiceRequestInput(
        request_id="config_demo_001",
        text=text,
    )
    payload = {
        "service_request": service_request.model_dump(mode="json"),
        "retrieved_references": [reference.model_dump(mode="json")],
    }
    return StructuredGenerationRequest(
        messages=[
            ChatMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False),
            )
        ],
        schema_name="service_case_result",
        json_schema={"type": "object"},
    )


def write_custom_configuration(directory: Path) -> None:
    """创建一组与默认配置不同的临时 YAML，用于证明配置会实际生效。"""

    (directory / "scenario.yaml").write_text(
        """scenario_id: scenic_service
input_channel: text
display_name: 自定义景区演示
event_matchers:
  - event_type: facility_fault
    facility_name: 自定义设备
    keyword_groups:
      - [自定义设备]
      - [坏了]
""",
        encoding="utf-8",
    )
    (directory / "risk_rules.yaml").write_text(
        """scenario_id: scenic_service
rules:
  - rule_id: custom_demo_risk
    keywords: [专用风险词]
    risk_level: high
    requires_human_review: true
    review_reason: high_risk
    risk_summary: 命中自定义演示风险词。
    review_note: 自定义风险规则要求人工复核。
""",
        encoding="utf-8",
    )
    (directory / "routing.yaml").write_text(
        """scenario_id: scenic_service
routes:
  - route_id: custom_demo_route
    event_type: facility_fault
    risk_level: low
    required_fields: [location]
    suggested_action: 使用自定义配置中的演示建议。
    knowledge_source_id: custom_source_001
    is_demo_action: true
""",
        encoding="utf-8",
    )


def test_default_configuration_exposes_supported_event_risk_and_route() -> None:
    """仓库默认 YAML 应能提供事件、风险和低风险演示建议。"""

    configuration = load_scenic_service_configuration()

    event_matcher = configuration.find_event_matcher("西门照明故障")
    risk_rule = configuration.find_high_risk_rule("游客摔倒，需要帮助")
    assert event_matcher is not None
    route = configuration.find_route(
        event_type=event_matcher.event_type,
        risk_level=RiskLevel.LOW,
    )

    assert event_matcher.facility_name == "照明设施"
    assert risk_rule is not None
    assert risk_rule.requires_human_review is True
    assert route is not None
    assert route.suggested_action == "创建演示性设施维护跟进建议。"


def test_demo_provider_uses_injected_yaml_configuration(tmp_path: Path) -> None:
    """注入临时配置后，事件名称、建议和风险词都必须来自 YAML。"""

    write_custom_configuration(tmp_path)
    configuration = load_scenic_service_configuration(tmp_path)
    provider = DemoLLMProvider(configuration=configuration)
    reference = KnowledgeReference(
        source_id="custom_source_001",
        source_title="自定义演示资料",
        source_path="data/custom.md",
        excerpt="仅用于配置回归测试。",
        relevance_score=1.0,
    )

    normal_response = provider.generate_json(
        build_provider_request("西门自定义设备坏了", reference)
    )
    normal_payload = json.loads(normal_response.content)

    assert normal_payload["entities"]["facility_name"] == "自定义设备"
    assert normal_payload["action_plan"][0]["suggested_action"] == (
        "使用自定义配置中的演示建议。"
    )
    assert normal_payload["review"]["requires_human_review"] is False

    risk_response = provider.generate_json(
        build_provider_request("西门自定义设备坏了，出现专用风险词", reference)
    )
    risk_payload = json.loads(risk_response.content)

    assert risk_payload["risk"]["level"] == "high"
    assert risk_payload["review"]["requires_human_review"] is True
    assert risk_payload["review"]["reasons"] == ["high_risk"]
    assert risk_payload["action_plan"] == []


def test_configuration_rejects_a_non_high_risk_rule(tmp_path: Path) -> None:
    """配置不能把命中风险词的案件错误降级为低风险。"""

    write_custom_configuration(tmp_path)
    risk_rules_path = tmp_path / "risk_rules.yaml"
    risk_rules_path.write_text(
        risk_rules_path.read_text(encoding="utf-8").replace(
            "risk_level: high", "risk_level: low"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScenarioConfigurationError, match="high risk"):
        load_scenic_service_configuration(tmp_path)
