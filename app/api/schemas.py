"""定义 FastAPI 接口接收的数据模型。"""

from pydantic import Field

from app.schemas.models import ImageAttachment, StrictSchema


class TriageApiRequest(StrictSchema):
    """HTTP 接口的宽松输入模型，图片附件为可选字段。"""

    # 不能在这里设为必填，否则 FastAPI 会在进入 Agent 前直接返回默认 422。
    request_id: str | None = Field(default=None, max_length=100)
    text: str | None = Field(default=None, max_length=2000)

    # 图片内容在请求期间临时存在；保存结果时只保留摘要，不保存 base64。
    image: ImageAttachment | None = None
