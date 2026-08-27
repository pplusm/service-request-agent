# 演示与录屏脚本

本脚本用于展示项目第一阶段的文本和可选图片分诊能力。全部输入都是演示内容，不代表真实
景区事件、真实人员信息或真实处置规则。

## 演示前准备

在项目根目录打开两个 PowerShell 窗口，并分别运行：

```powershell
conda activate service-request-agent
Set-Location E:\project\agent\service-request-agent
python -m uvicorn app.api.main:app --reload
```

```powershell
conda activate service-request-agent
Set-Location E:\project\agent\service-request-agent
python -m streamlit run app\ui\streamlit_app.py
```

浏览器打开 `http://127.0.0.1:8501`。确认左侧“FastAPI 服务地址”为
`http://127.0.0.1:8000`。本次演示默认使用本地文本和视觉 `demo` 模型，不需要输入或展示
API Key。

## 建议录制顺序

1. 打开“提交分诊”标签页，说明此阶段接收景区服务文本，可选上传一张图片；结果始终是
   经过 Pydantic 校验的 JSON。
2. 输入案件编号 `demo_facility_001`，并输入演示文本 `西门照明故障`，点击“提交分诊”。
   展示低风险、知识库命中、知识来源和演示建议。说明这些建议只来自本项目本地演示资料，
   不是真实景区处置指令。
3. 回到“提交分诊”，输入案件编号 `demo_image_001`，并输入演示文本 `西门照明故障`，
   上传一张 JPG、PNG 或 WebP 演示图片后提交。展示“图片观察”和“图文融合判断”。说明默认
   视觉 demo 只确认图片已接收，不分析真实像素，所以融合状态是 `not_assessed`，该案件必须
   人工复核；结果不会保存 Base64 图片内容，也不能把本次演示说成真实识图。
4. 回到“提交分诊”，输入案件编号 `demo_review_001`，并输入演示文本
   `东门附近有游客突然晕倒，需要帮助`，点击“提交分诊”。展示页面中“必须人工复核”的
   警告，说明高风险案例不会自动给出最终处置决定。
5. 打开“案件历史”标签页，展示三条已校验的本地记录；说明数据库只保存结果 JSON，
   用于本机演示。
6. 打开“待人工复核”标签页，展示高风险案例仍在队列中；说明项目不会自动把它标记为
   已复核，也不模拟真实调度。
7. 可选地打开 `http://127.0.0.1:8000/docs`，展示 FastAPI 的 `POST /api/v1/triage`、
   `GET /api/v1/case-history` 和 `GET /api/v1/review-queue` 接口。

## API 图片请求示例

在 FastAPI 文档页的 `POST /api/v1/triage` 中，可以使用下面的演示 JSON。这里的 Base64
只是字符串 `demo-image` 的编码，不是可供识别的真实照片；它用于展示输入契约和安全摘要：

```json
{
  "request_id": "docs_image_001",
  "text": "西门照明故障",
  "image": {
    "media_type": "image/png",
    "data_base64": "ZGVtby1pbWFnZQ==",
    "filename": "demo.png"
  }
}
```

返回结果中的 `image` 只包含 `media_type`、`size_bytes`、`sha256` 和 `filename`；
`vision_observation.is_demo_observation` 为 `true` 时，表示当前仍使用本地演示视觉模型。此时
`multimodal_fusion.status` 会是 `not_assessed`，并强制进入人工复核。

## 图文融合测试样例

默认 Demo 不读取真实图片像素，因此页面只能演示“未评估”的安全结果。项目仍提供 10 条以上
不依赖外部 API 的图文融合单元测试，覆盖图文一致、Demo 未评估、低置信度、不确定说明、证据
缺失，以及地点、设施和状态冲突。运行：

```powershell
conda activate service-request-agent
Set-Location E:\project\agent\service-request-agent
python -m pytest tests\agent\test_multimodal_fusion.py
```

这些测试使用固定的结构化模拟视觉观察，不代表对真实照片的识别准确率。

## 录制时不要展示的内容

- API Key、`.env` 文件、终端环境变量的密钥值或第三方模型后台页面。
- 真实照片、Base64 原文或任何包含个人信息的图片。
- 真实姓名、电话、位置、景区业务记录或任何真实个人数据。
- “已经自动处置”“真实角色已接单”“准确率多少”等不属于本项目演示能力的表述。
