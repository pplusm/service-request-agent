"""加载并校验景区服务演示场景的 YAML 配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.schemas.models import EventType, InputChannel, ReviewReason, RiskLevel, ScenarioId


# 该目录保存第一阶段景区服务场景的演示配置，不含真实业务规则。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIRECTORY = PROJECT_ROOT / "scenarios" / "scenic_service"


class ScenarioConfigurationError(ValueError):
    """YAML 文件缺失、格式错误或不符合数据契约时抛出的异常。"""


class ConfigSchema(BaseModel):
    """所有场景配置共用的严格校验规则。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EventMatcher(ConfigSchema):
    """一条可由演示模型识别的事件匹配规则。"""

    event_type: EventType
    facility_name: str = Field(min_length=1, max_length=100)

    # 每个关键词组至少命中一个词；所有组都命中时，才算识别为该事件。
    keyword_groups: list[list[str]] = Field(min_length=1, max_length=10)

    @field_validator("keyword_groups")
    @classmethod
    def validate_keyword_groups(cls, value: list[list[str]]) -> list[list[str]]:
        """拒绝空关键词组，避免配置意外匹配所有输入。"""

        if any(not group or any(not keyword.strip() for keyword in group) for group in value):
            raise ValueError("every keyword group must contain non-empty keywords")
        return value

    def matches(self, text: str) -> bool:
        """判断文本是否满足本条演示事件的全部关键词组。"""

        normalized_text = text.strip()
        return bool(normalized_text) and all(
            any(keyword in normalized_text for keyword in group)
            for group in self.keyword_groups
        )


class ScenarioConfig(ConfigSchema):
    """场景可识别事件的演示配置。"""

    scenario_id: ScenarioId
    input_channel: InputChannel = InputChannel.TEXT
    display_name: str = Field(min_length=1, max_length=100)
    event_matchers: list[EventMatcher] = Field(min_length=1, max_length=30)


class RiskRule(ConfigSchema):
    """命中后必须转人工的演示风险规则。"""

    rule_id: str = Field(min_length=1, max_length=100)
    keywords: list[str] = Field(min_length=1, max_length=100)
    risk_level: RiskLevel
    requires_human_review: Literal[True]
    review_reason: ReviewReason
    risk_summary: str = Field(min_length=1, max_length=500)
    review_note: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_high_risk_rule(self) -> "RiskRule":
        """第一阶段的风险词规则只能提升为高风险并强制人工复核。"""

        if self.risk_level != RiskLevel.HIGH:
            raise ValueError("demo risk rules must use the high risk level")
        if self.review_reason != ReviewReason.HIGH_RISK:
            raise ValueError("demo risk rules must use the high_risk review reason")
        if any(not keyword.strip() for keyword in self.keywords):
            raise ValueError("risk rule keywords must not be empty")
        return self

    def matches(self, text: str) -> bool:
        """判断输入是否含有任一演示高风险关键词。"""

        normalized_text = text.strip()
        return bool(normalized_text) and any(
            keyword in normalized_text for keyword in self.keywords
        )


class RiskRulesConfig(ConfigSchema):
    """景区场景全部演示风险规则的容器。"""

    scenario_id: ScenarioId
    rules: list[RiskRule] = Field(min_length=1, max_length=30)


class DemonstrationRoute(ConfigSchema):
    """低风险演示案件可使用的建议配置，不代表真实处置路由。"""

    route_id: str = Field(min_length=1, max_length=100)
    event_type: EventType
    risk_level: RiskLevel
    required_fields: list[
        Literal[
            "location",
            "facility_name",
            "visitor_condition",
            "estimated_affected_count",
            "event_time_description",
        ]
    ] = Field(min_length=1, max_length=5)
    suggested_action: str = Field(min_length=1, max_length=500)
    knowledge_source_id: str = Field(min_length=1, max_length=100)
    is_demo_action: Literal[True]

    @model_validator(mode="after")
    def validate_low_risk_route(self) -> "DemonstrationRoute":
        """禁止通过配置为中高风险案件生成自动演示建议。"""

        if self.risk_level != RiskLevel.LOW:
            raise ValueError("demonstration routes must only target low risk")
        return self


class RoutingConfig(ConfigSchema):
    """低风险演示建议的匹配配置。"""

    scenario_id: ScenarioId
    routes: list[DemonstrationRoute] = Field(min_length=1, max_length=30)


class ScenicServiceConfiguration(ConfigSchema):
    """将三个独立 YAML 文件组成一份经过校验的景区演示配置。"""

    scenario: ScenarioConfig
    risk_rules: RiskRulesConfig
    routing: RoutingConfig

    @model_validator(mode="after")
    def validate_shared_scenario_id(self) -> "ScenicServiceConfiguration":
        """确保不同文件不会误拼接成不同场景的规则。"""

        scenario_id = self.scenario.scenario_id
        if self.risk_rules.scenario_id != scenario_id:
            raise ValueError("risk_rules scenario_id must match scenario scenario_id")
        if self.routing.scenario_id != scenario_id:
            raise ValueError("routing scenario_id must match scenario scenario_id")
        return self

    def find_event_matcher(self, text: str) -> EventMatcher | None:
        """按 YAML 中的顺序返回第一条匹配的演示事件规则。"""

        return next(
            (matcher for matcher in self.scenario.event_matchers if matcher.matches(text)),
            None,
        )

    def find_high_risk_rule(self, text: str) -> RiskRule | None:
        """按 YAML 中的顺序返回第一条命中的高风险规则。"""

        return next(
            (rule for rule in self.risk_rules.rules if rule.matches(text)),
            None,
        )

    def find_route(
        self,
        *,
        event_type: EventType,
        risk_level: RiskLevel,
    ) -> DemonstrationRoute | None:
        """按事件类型和风险等级查找可使用的低风险演示建议。"""

        return next(
            (
                route
                for route in self.routing.routes
                if route.event_type == event_type and route.risk_level == risk_level
            ),
            None,
        )


def load_scenic_service_configuration(
    config_directory: Path | None = None,
) -> ScenicServiceConfiguration:
    """安全读取三个 YAML 文件，并返回可供 Agent 使用的严格配置对象。"""

    directory = config_directory or DEFAULT_SCENARIO_DIRECTORY
    try:
        return ScenicServiceConfiguration(
            scenario=_load_yaml_mapping(directory / "scenario.yaml"),
            risk_rules=_load_yaml_mapping(directory / "risk_rules.yaml"),
            routing=_load_yaml_mapping(directory / "routing.yaml"),
        )
    except ValidationError as error:
        raise ScenarioConfigurationError(
            f"scenic service configuration is invalid: {error}"
        ) from error


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    """使用 safe_load 读取 YAML，且只接受顶层对象，避免不受控配置。"""

    if not path.is_file():
        raise ScenarioConfigurationError(f"configuration file does not exist: {path}")

    try:
        raw_content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScenarioConfigurationError(
            f"failed to read configuration file {path}: {error}"
        ) from error

    if not isinstance(raw_content, dict):
        raise ScenarioConfigurationError(
            f"configuration file must contain a YAML object: {path}"
        )
    return raw_content
