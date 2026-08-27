"""提供不联网、免费的确定性视觉演示模型。"""

from __future__ import annotations

import json

from app.schemas.models import VisionObservation
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProvider,
    VisionProviderError,
    VisionResponse,
)


class DemoVisionProvider(VisionProvider):
    """只确认图片已接收的本地演示提供方，不假装具备真实像素识别能力。"""

    def __init__(
        self,
        *,
        simulate_invalid_output: bool = False,
        error_message: str | None = None,
    ) -> None:
        # 这两个开关用于演示视觉调用失败和输出解析失败时的安全兜底。
        self._simulate_invalid_output = simulate_invalid_output
        self._error_message = error_message
        self.requests: list[VisionExtractionRequest] = []

    def generate_json(self, request: VisionExtractionRequest) -> VisionResponse:
        """返回明确标注为演示的结构化观察，不推断图片中的真实内容。"""

        self.requests.append(request.model_copy(deep=True))
        if self._error_message is not None:
            raise VisionProviderError(self._error_message)
        if self._simulate_invalid_output:
            return VisionResponse(
                content="这不是合法的视觉 JSON",
                provider_name="local_demo_vision",
                model_name="deterministic-demo-vision-v1",
            )

        observation = VisionObservation(
            description=(
                f"本地演示视觉提供方已接收一张 {request.image.media_type} 图片，"
                "但当前未启用真实像素识别。"
            ),
            objects=[],
            visible_text=[],
            location_hint=None,
            facility_hint=None,
            hazard_signals=[],
            uncertainty_notes=["演示模型未分析图片像素，不能据此确认现场事实。"],
            confidence=0.0,
            is_demo_observation=True,
        )
        return VisionResponse(
            content=json.dumps(observation.model_dump(mode="json"), ensure_ascii=False),
            provider_name="local_demo_vision",
            model_name="deterministic-demo-vision-v1",
        )

