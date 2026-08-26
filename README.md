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

## 演示场景配置

景区服务的演示规则不再直接写在 Python 判断语句中，统一放在以下文件：

- `scenarios/scenic_service/scenario.yaml`：当前可识别的演示事件和关键词组合。
- `scenarios/scenic_service/risk_rules.yaml`：命中后必须升级人工复核的演示高风险词。
- `scenarios/scenic_service/routing.yaml`：低风险设施故障可展示的演示建议及其知识来源。

这些内容全部是项目演示配置，不代表真实景区角色、调度路径或服务时限。配置文件会在
FastAPI 启动时经 Pydantic 校验；修改 YAML 后，请重启 FastAPI 服务再测试。

## 测试

运行全部测试：

```powershell
python -m pytest
```

只验证场景配置及其与模拟模型的连接：

```powershell
python -m pytest tests\rules\test_scenic_service_config.py
```
