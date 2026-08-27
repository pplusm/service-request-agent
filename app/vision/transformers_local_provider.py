"""通过 Transformers 在本机 GPU 上运行 Qwen2.5-VL 的视觉提供方。"""

from __future__ import annotations

import base64
import binascii
import os
import re
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from pydantic import Field, ValidationError, field_validator

from app.llm.provider import ProviderSchema
from app.schemas.models import ImageAttachment
from app.vision.provider import (
    VisionExtractionRequest,
    VisionProvider,
    VisionProviderError,
    VisionResponse,
)


# Qwen2.5-VL 的视觉 token 以 28 x 28 像素为一个单位；512 个 token 适合 8 GB 显存演示。
_DEFAULT_MAX_PIXELS = 512 * 28 * 28
_MAX_ALLOWED_PIXELS = 2_048 * 28 * 28


class LocalQwenVisionSettings(ProviderSchema):
    """本地 Qwen2.5-VL 视觉推理的最小运行配置。"""

    # 模型权重保存在项目仓库外，避免将数 GB 文件误提交到 Git。
    model_path: Path
    # 限制图片分辨率对应的视觉 token 数，避免单张图片占满 8 GB 显存。
    max_pixels: int = Field(
        default=_DEFAULT_MAX_PIXELS,
        ge=28 * 28,
        le=_MAX_ALLOWED_PIXELS,
    )
    # 限制模型回答长度，防止异常输出占用过多显存或进入案件历史。
    max_new_tokens: int = Field(default=512, ge=32, le=1_024)

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: Path) -> Path:
        """只接受已经存在的本地模型目录。"""

        normalized_path = value.expanduser()
        if not normalized_path.is_dir():
            raise ValueError("model_path 必须是已存在的本地模型目录")
        if len(normalized_path.name) > 200:
            raise ValueError("model_path 的最后一级目录名不能超过 200 个字符")
        return normalized_path.resolve()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "LocalQwenVisionSettings":
        """从环境变量读取配置，缺少模型目录时拒绝静默回退。"""

        environment = os.environ if environ is None else environ
        model_path = environment.get("LOCAL_QWEN_VISION_MODEL_PATH", "").strip()
        if not model_path:
            raise VisionProviderError(
                "缺少本地 Qwen 视觉模型路径配置："
                "LOCAL_QWEN_VISION_MODEL_PATH。"
            )

        try:
            return cls(
                model_path=model_path,
                max_pixels=environment.get(
                    "LOCAL_QWEN_VISION_MAX_PIXELS",
                    str(_DEFAULT_MAX_PIXELS),
                ),
                max_new_tokens=environment.get(
                    "LOCAL_QWEN_VISION_MAX_NEW_TOKENS",
                    "512",
                ),
            )
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: "
                f"{item['msg']}"
                for item in error.errors(include_url=False)
            )
            raise VisionProviderError(
                "本地 Qwen 视觉模型配置无效：" + details
            ) from error


class LocalQwenVisionBackend(Protocol):
    """隔离重量级 Transformers 调用，便于单元测试使用内存替身。"""

    def load(
        self,
        settings: LocalQwenVisionSettings,
    ) -> tuple[Any, Any]:
        """加载并返回模型与处理器。"""

    def generate(
        self,
        *,
        model: Any,
        processor: Any,
        image: Any,
        request: VisionExtractionRequest,
        settings: LocalQwenVisionSettings,
    ) -> str:
        """根据内存中的图片生成尚未经过 Pydantic 校验的文本。"""


ImageDecoder = Callable[[ImageAttachment], Any]


class TransformersLocalQwenBackend:
    """真实的 Qwen2.5-VL Transformers 推理实现。"""

    def load(
        self,
        settings: LocalQwenVisionSettings,
    ) -> tuple[Any, Any]:
        """仅在第一次图片请求时导入依赖并加载 4-bit 模型。"""

        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as error:
            raise VisionProviderError(
                "本地 Qwen 视觉依赖未安装，请安装 torch、transformers、"
                "accelerate、bitsandbytes 和 Pillow。"
            ) from error

        if not torch.cuda.is_available():
            # 3B 视觉模型在 CPU 上不适合交互式 Streamlit 页面，不能悄悄降级运行。
            raise VisionProviderError(
                "本地 Qwen 视觉模型未检测到可用 CUDA 显卡，"
                "当前不会回退到缓慢的 CPU 推理。"
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(settings.model_path),
            local_files_only=True,
            quantization_config=quantization_config,
            device_map="auto",
            # 不依赖 FlashAttention，降低 Windows 环境的额外安装要求。
            attn_implementation="sdpa",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            str(settings.model_path),
            local_files_only=True,
            max_pixels=settings.max_pixels,
            # 固定使用原检查点对应的慢速处理器，避免版本升级后静默改变预处理结果。
            use_fast=False,
        )
        return model, processor

    def generate(
        self,
        *,
        model: Any,
        processor: Any,
        image: Any,
        request: VisionExtractionRequest,
        settings: LocalQwenVisionSettings,
    ) -> str:
        """把内存图片交给 Qwen，并返回模型生成的原始文本。"""

        try:
            import torch
        except ImportError as error:
            raise VisionProviderError("本地 Qwen 视觉依赖 torch 未安装。") from error

        messages = [
            {
                "role": "system",
                "content": _build_qwen_system_instruction(request),
            },
            {
                "role": "user",
                "content": [
                    # 这里直接传递 Pillow 图片对象，图片 Base64 不会变成临时文件。
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        contents = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not contents or not isinstance(contents[0], str):
            raise VisionProviderError("本地 Qwen 视觉模型未返回可读取的文本。")
        return contents[0]


class TransformersLocalVisionProvider(VisionProvider):
    """延迟加载的本地 Qwen2.5-VL 提供方，保留现有 Pydantic 安全边界。"""

    def __init__(
        self,
        settings: LocalQwenVisionSettings,
        *,
        backend: LocalQwenVisionBackend | None = None,
        image_decoder: ImageDecoder | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend or TransformersLocalQwenBackend()
        self._image_decoder = image_decoder or _decode_image_in_memory
        self._model: Any | None = None
        self._processor: Any | None = None
        # 同一 FastAPI 进程只加载一次模型，防止并发首请求重复占用显存。
        self._model_load_lock = Lock()
        # 8 GB 显存不适合同时执行多次 generate，因此把视觉推理串行化。
        self._inference_lock = Lock()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "TransformersLocalVisionProvider":
        """供视觉 Provider 工厂从环境变量创建实例。"""

        return cls(LocalQwenVisionSettings.from_environment(environ))

    def generate_json(self, request: VisionExtractionRequest) -> VisionResponse:
        """生成原始 JSON 文本；最终字段校验仍由 LangGraph 节点完成。"""

        if request.schema_name != "vision_observation":
            raise VisionProviderError(
                "本地 Qwen 视觉提供方目前只支持 vision_observation 输出。"
            )

        try:
            image = self._image_decoder(request.image)
        except VisionProviderError:
            raise
        except Exception as error:
            # 不能把异常详情回显，图片解码器可能在消息中包含不应记录的内容。
            raise VisionProviderError("本地 Qwen 无法解码图片内容。") from error

        with self._inference_lock:
            model, processor = self._get_loaded_components()
            try:
                content = self._backend.generate(
                    model=model,
                    processor=processor,
                    image=image,
                    request=request,
                    settings=self._settings,
                )
            except VisionProviderError:
                raise
            except Exception as error:
                # 模型异常统一交给 Agent 转人工复核，不能伪造或回显图片数据。
                raise VisionProviderError(
                    "本地 Qwen 视觉模型推理失败，已停止本次图片自动判断。"
                ) from error

        if not isinstance(content, str):
            raise VisionProviderError("本地 Qwen 视觉模型返回的内容不是字符串。")
        if len(content) > 50_000:
            raise VisionProviderError("本地 Qwen 视觉模型返回内容超过安全长度限制。")

        return VisionResponse(
            content=_unwrap_single_json_code_fence(content),
            provider_name="transformers_local_vision",
            model_name=self._settings.model_path.name,
        )

    def _get_loaded_components(self) -> tuple[Any, Any]:
        """在第一次调用时加载模型，之后复用同一份显存中的权重。"""

        with self._model_load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor

            try:
                model, processor = self._backend.load(self._settings)
            except VisionProviderError:
                raise
            except Exception as error:
                raise VisionProviderError(
                    "本地 Qwen 视觉模型加载失败，请检查模型目录和显卡运行环境。"
                ) from error

            if model is None or processor is None:
                raise VisionProviderError(
                    "本地 Qwen 视觉模型加载未返回完整模型和处理器。"
                )
            self._model = model
            self._processor = processor
            return model, processor


def _decode_image_in_memory(image: ImageAttachment) -> Any:
    """把 Base64 图片只在内存中解码为 RGB Pillow 对象。"""

    try:
        from PIL import Image
    except ImportError as error:
        raise VisionProviderError(
            "本地 Qwen 视觉依赖 Pillow 未安装，无法读取图片。"
        ) from error

    try:
        raw_image = base64.b64decode(image.data_base64, validate=True)
        with Image.open(BytesIO(raw_image)) as opened_image:
            # copy/convert 后即可关闭 BytesIO 与原始图片对象，不保留文件句柄。
            return opened_image.convert("RGB")
    except (OSError, ValueError, binascii.Error) as error:
        raise VisionProviderError("本地 Qwen 无法解码图片内容。") from error


def _build_qwen_system_instruction(request: VisionExtractionRequest) -> str:
    """给 Qwen 明确字段模板，避免它把 JSON Schema 元数据误当成输出字段。"""

    # 这里故意不直接粘贴 Pydantic 的完整 JSON Schema；小模型可能会错误输出 maxItems 等约束词。
    field_template = """{
  \"description\": \"short observable description\",
  \"objects\": [\"observable object\"],
  \"visible_text\": [\"literal text visible in image\"],
  \"location_hint\": \"one location hint or null\",
  \"facility_hint\": \"one facility hint or null\",
  \"hazard_signals\": [\"observable hazard signal\"],
  \"uncertainty_notes\": [\"what cannot be confirmed\"],
  \"confidence\": 0.0,
  \"is_demo_observation\": false
}"""
    return (
        "You inspect an image for a scenic-service request. Return exactly one JSON "
        "object and nothing else. Do not use Markdown, code fences, or explanation. "
        "Only report observable image facts. Do not invent service rules, handling "
        "roles, precise locations, or hidden facts. visible_text must contain only "
        "literal text visible in the image; never copy text from the user task. "
        "The output must have exactly the nine keys shown below. location_hint and "
        "facility_hint must each be one string or null, never an array. Do not add "
        "JSON Schema keywords or extra keys. Use empty arrays, null, or 0.0 when "
        "unknown. confidence must be between 0 and 1. is_demo_observation must be "
        "false. You may use Chinese or the image's original language for field values.\n"
        "Field template:\n"
        f"{field_template}\n"
        "Task:\n"
        f"{request.prompt}"
    )


def _unwrap_single_json_code_fence(content: str) -> str:
    """仅移除完整包裹 JSON 的单个 Markdown 代码块，不修补任何模型字段。"""

    stripped_content = content.strip()
    match = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(?P<json>.*?)(?:\r?\n)?```",
        stripped_content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        # 非严格代码块或夹杂额外文字时保持原样，让后续 Pydantic 校验失败并转人工复核。
        return content
    return match.group("json").strip()
