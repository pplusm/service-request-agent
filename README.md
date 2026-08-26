# 多模态服务诉求分诊与处置 Agent

景区服务文本诉求分诊与处置辅助决策 Demo。

## MVP 范围

- 仅支持景区服务文本输入。
- 使用 Pydantic 校验 JSON 输出。
- 高风险、知识库未命中、字段缺失与模型解析失败均转人工复核。
- 只使用演示数据和可追溯知识引用。
- 默认使用免费的本地确定性演示模型，不调用外部 API。

## 暂不实现

- 图片输入。
- 真实业务系统、真实个人数据和线上调度。
- 高风险事件的自动最终决策。

## 终端演示

在已激活的 `service-request-agent` Conda 环境中运行：

```powershell
python scenarios\scenic_service\run_demo_agent.py "西门照明故障"
```

## 本地 API

安装项目依赖后启动 FastAPI：

```powershell
python -m uvicorn app.api.main:app --reload
```

然后在浏览器打开 `http://127.0.0.1:8000/docs`，使用
`POST /api/v1/triage` 接口测试文本诉求。

## 本地操作页面

先保持上方 FastAPI 服务正在运行。然后在项目根目录打开第二个、已激活
`service-request-agent` 环境的 PowerShell 窗口，运行：

```powershell
python -m streamlit run app\ui\streamlit_app.py
```

浏览器打开 `http://localhost:8501`。在页面中输入案件编号和景区服务诉求，
页面会调用本地 API，并只展示再次通过 Pydantic 校验的案件 JSON。
