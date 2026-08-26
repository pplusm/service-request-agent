# Demo crowd congestion knowledge

> **仅供项目演示。** 本文用于 `service-request-agent` 的本地检索测试，
> 不是实际景区服务规范，不指定真实岗位、处置流程或服务时限。

- Source ID: `demo_crowd_001`
- Scenario: `scenic_service`
- Intended event type: `crowd_congestion`
- Citation path: `data/scenic_service/knowledge/demo_crowd_congestion.md`
- Retrieval keywords: `客流拥挤`, `入口拥堵`, `通道拥挤`, `人流较多`

## 可检索演示主题

本文可用于检索包含“客流拥挤”“入口拥堵”“通道拥挤”或“人流较多”等描述的文本。
这些词只帮助演示 RAG 如何找回资料，不能单独证明事件等级或处理结论。

## 当前 Agent 边界

第一阶段尚未为客流拥挤配置自动分类、自动路由或自动处置建议。即使检索到本文，
Agent 也必须保守地转人工复核，由人员结合现场信息判断后续处理。
