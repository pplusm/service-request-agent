# 演示与录屏脚本

本脚本用于展示项目第一阶段的文本分诊能力。全部输入都是演示文本，不代表真实景区事件、
真实人员信息或真实处置规则。

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
`http://127.0.0.1:8000`。本次演示默认使用本地 `demo` 模型，不需要输入或展示 API Key。

## 建议录制顺序

1. 打开“提交分诊”标签页，说明此阶段只接收景区服务文本，结果始终是经过 Pydantic
   校验的 JSON。
2. 输入案件编号 `demo_facility_001`，并输入演示文本 `西门照明故障`，点击“提交分诊”。
   展示低风险、知识库命中、知识来源和演示建议。说明这些建议只来自本项目本地演示资料，
   不是真实景区处置指令。
3. 回到“提交分诊”，输入案件编号 `demo_review_001`，并输入演示文本
   `东门附近有游客突然晕倒，需要帮助`，点击“提交分诊”。展示页面中“必须人工复核”的
   警告，说明高风险案例不会自动给出最终处置决定。
4. 打开“案件历史”标签页，展示两条已校验的本地记录；说明数据库只保存结果 JSON，
   用于本机演示。
5. 打开“待人工复核”标签页，展示高风险案例仍在队列中；说明项目不会自动把它标记为
   已复核，也不模拟真实调度。
6. 可选地打开 `http://127.0.0.1:8000/docs`，展示 FastAPI 的 `POST /api/v1/triage`、
   `GET /api/v1/case-history` 和 `GET /api/v1/review-queue` 接口。

## 录制时不要展示的内容

- API Key、`.env` 文件、终端环境变量的密钥值或第三方模型后台页面。
- 真实姓名、电话、位置、景区业务记录或任何真实个人数据。
- “已经自动处置”“真实角色已接单”“准确率多少”等不属于本项目演示能力的表述。
