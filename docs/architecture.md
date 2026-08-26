# 系统架构

## 架构总览

```mermaid
flowchart LR
    terminal["终端演示"] --> api["FastAPI\n/api/v1/triage"]
    ui["Streamlit 操作页面"] --> api

    api --> graph["LangGraph 工作流\n检索 -> 生成或安全兜底 -> 解析"]
    graph --> rag["ChromaDB\n本地演示知识库"]
    graph --> rules["YAML 演示规则\n场景、风险、建议"]
    graph --> factory["模型提供方工厂"]
    factory --> demo["默认：本地 demo 模型"]
    factory -->|可选| compatible["OpenAI-compatible 提供方"]

    graph --> validation["Pydantic 输出校验\n统一 JSON 契约"]
    validation --> response["API / 页面展示"]
    validation --> history["SQLite 本地案件历史\n待人工复核队列"]
```

## 一次文本诉求如何流转

1. 用户在终端、FastAPI 文档页或 Streamlit 页面提交一条景区服务文本。
2. FastAPI 校验基础请求字段；请求体缺失、字段为空或格式不正确时，仍返回符合
   Pydantic 契约的“必须人工复核”结果。
3. LangGraph 先从 ChromaDB 检索本地演示知识。知识未命中或知识库发生异常时，工作流
   不调用模型，直接转人工复核。
4. 知识命中后，模型提供方工厂选择默认的本地 `demo` 模型；将来可通过环境变量替换为
   OpenAI-compatible API。无论提供方是什么，模型只能被要求返回 JSON。
5. 模型输出会经过 Pydantic 解析和演示风险规则兜底。输出解析失败、调用失败、高风险词
   命中或信息不足时，结果都会设置 `requires_human_review=true`。
6. 已通过 Pydantic 校验的结果才会作为 API JSON 返回，并写入本机 SQLite 案件历史。
   “待人工复核”页面只读取其中仍需要人工处理的记录。

## 组件职责

| 组件 | 职责 |
| --- | --- |
| FastAPI | 接收 HTTP 请求、统一返回经过 Pydantic 校验的 JSON、保存本地历史。 |
| LangGraph | 编排检索、模型生成、解析和安全兜底这几个处理节点。 |
| ChromaDB | 保存并检索 `data/scenic_service/knowledge` 中的本地演示资料。 |
| YAML 配置 | 保存可识别场景、高风险词和低风险设施故障的演示建议。 |
| 模型提供方 | 封装本地 demo 模型与可替换的 OpenAI-compatible `/chat/completions` 接口。 |
| Pydantic | 定义输入、输出、知识来源、人工复核和历史记录的数据结构。 |
| SQLite | 仅保存已经校验过的本地演示结果，不单独保存原始请求体。 |
| Streamlit | 提供提交分诊、查看历史和查看人工复核队列的本地页面。 |

## 安全边界

- 当前项目仅实现文本输入；图片和其他多模态输入尚未实现。
- 所有知识、规则和建议都是演示数据，不对应真实景区的服务标准、角色、时限或调度流程。
- 项目不宣称模型准确率，也不会自动完成高风险事件的最终处置。
- 默认使用无需付费 API 的本地 demo 模型。外部模型密钥只应保存在本机环境变量中，不能
  写入代码、Git 仓库、文档或截图。
