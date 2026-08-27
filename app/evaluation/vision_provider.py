"""提供仅供评测使用的确定性视觉观察，不读取真实图片像素。"""

from __future__ import annotations

import json

from app.evaluation.models import VisionBehavior
from app.schemas.models import VisionObservation
from app.vision.demo_provider import DemoVisionProvider
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProvider,
    VisionResponse,
)


class EvaluationVisionProvider(VisionProvider):
    """按案例声明返回固定视觉观察，专门验证图文融合和人工复核流程。

    这个类不检查图片字节，也不声称能够从 PNG 中识别现场事实。图片夹具只用于
    验证输入传输和元数据处理；这里的观察值是评测者预先写好的确定性模拟数据。
    """

    def __init__(self, behavior: VisionBehavior) -> None:
        self._behavior = behavior
        self.requests: list[VisionExtractionRequest] = []
        self._demo_provider = DemoVisionProvider()

    def generate_json(self, request: VisionExtractionRequest) -> VisionResponse:
        """返回与指定评测行为对应的 Pydantic 可校验 JSON 文本。"""

        self.requests.append(request.model_copy(deep=True))
        if self._behavior == VisionBehavior.DEMO:
            # 保持产品默认 Demo 的真实行为，明确返回“未分析像素”。
            return self._demo_provider.generate_json(request)

        observation = _build_observation(self._behavior)
        return VisionResponse(
            content=json.dumps(observation.model_dump(mode="json"), ensure_ascii=False),
            provider_name="evaluation_deterministic_vision",
            model_name=f"fixture-{self._behavior.value}-v1",
        )


def _build_observation(behavior: VisionBehavior) -> VisionObservation:
    """集中维护四种公开夹具对应的模拟观察，方便审阅评测含义。"""

    if behavior == VisionBehavior.LIGHTING_FAULT:
        return VisionObservation(
            description="西门的路灯不亮。",
            objects=["路灯"],
            location_hint="西门",
            facility_hint="照明设施",
            confidence=0.95,
        )
    if behavior == VisionBehavior.LIGHTING_NORMAL:
        return VisionObservation(
            description="西门的路灯正常。",
            objects=["路灯"],
            location_hint="西门",
            facility_hint="照明设施",
            confidence=0.95,
        )
    if behavior == VisionBehavior.LOW_CONFIDENCE:
        return VisionObservation(
            description="画面较模糊，疑似有照明设施。",
            objects=["模糊的照明设施"],
            facility_hint="照明设施",
            uncertainty_notes=["图片细节不足，无法确认设施状态或地点。"],
            confidence=0.20,
        )
    raise ValueError(f"unsupported evaluation vision behavior: {behavior}")
