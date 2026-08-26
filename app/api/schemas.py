"""定义 FastAPI 接口接收的数据模型。"""

from pydantic import Field

from app.schemas.models import StrictSchema


class TriageApiRequest(StrictSchema):
    """HTTP 接口的宽松输入模型，字段缺失由业务层转人工复核。"""

    # 不能在这里设为必填，否则 FastAPI 会在进入 Agent 前直接返回默认 422。
    request_id: str | None = Field(default=None, max_length=100)
    text: str | None = Field(default=None, max_length=2000)
