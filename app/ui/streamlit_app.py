"""景区服务诉求分诊与处置 Agent 的 Streamlit 演示页面。"""

from __future__ import annotations

import os
from html import escape
from datetime import datetime

import streamlit as st

from app.schemas.models import ServiceCaseResult
from app.ui.api_client import TriageApiClientError, submit_triage_request


# 可通过环境变量修改地址，默认连接同一台电脑上的 FastAPI 服务。
DEFAULT_API_BASE_URL = os.getenv("TRIAGE_API_BASE_URL", "http://127.0.0.1:8000")


def _render_case_result(result: ServiceCaseResult) -> None:
    """以便于演示的方式展示已经通过 Pydantic 校验的案件结果。"""

    st.subheader("分诊结果")
    st.caption("案件编号")
    # 案件编号来自用户输入，先转义再用可换行样式展示，避免长编号溢出页面。
    st.markdown(
        "<div style='font-family: monospace; overflow-wrap: anywhere; "
        "word-break: break-word;'>"
        f"{escape(result.request_id)}"
        "</div>",
        unsafe_allow_html=True,
    )

    first_column, second_column = st.columns(2)
    first_column.metric("风险等级", result.risk.level.value)
    second_column.metric(
        "知识库命中",
        "是" if result.diagnostics.knowledge_hit else "否",
    )

    if result.review.requires_human_review:
        # 强制复核的结果使用醒目提示，避免用户把它误当作自动处置结论。
        reasons = "、".join(reason.value for reason in result.review.reasons)
        st.warning(f"该案件必须人工复核。原因：{reasons}")
    else:
        st.info("以下内容仅基于本项目演示资料生成，不代表真实景区处置指令。")

    st.subheader("识别信息")
    st.json(result.entities.model_dump(mode="json"))

    st.subheader("演示建议")
    if result.action_plan:
        for item in result.action_plan:
            st.write(f"{item.step}. {item.suggested_action}")
            st.caption(f"演示资料来源：{', '.join(item.knowledge_source_ids)}")
    else:
        st.write("当前没有可展示的演示建议。")

    st.subheader("知识来源")
    if result.knowledge_references:
        for reference in result.knowledge_references:
            st.write(f"{reference.source_id}: {reference.source_title}")
            st.caption(f"{reference.source_path} | {reference.excerpt}")
    else:
        st.write("本次未命中本地演示资料。")

    # 保留完整 JSON，方便答辩或测试时确认输出符合既定数据模型。
    with st.expander("完整 Pydantic JSON"):
        st.json(result.model_dump(mode="json"))


def main() -> None:
    """创建页面表单；页面只发起请求，不重复实现 Agent 判断逻辑。"""

    st.set_page_config(page_title="景区服务诉求分诊", layout="wide")
    st.header("景区服务诉求分诊")
    st.caption("多模态服务诉求分诊与处置 Agent · 第一阶段：景区服务文本演示")

    # 仅在本次浏览器会话首次加载页面时生成编号，之后允许用户稳定地手动修改。
    if "triage_request_id" not in st.session_state:
        st.session_state.triage_request_id = (
            f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    with st.sidebar:
        st.subheader("连接设置")
        api_base_url = st.text_input("FastAPI 服务地址", value=DEFAULT_API_BASE_URL)

    with st.form("triage_form"):
        request_id = st.text_input(
            "案件编号",
            key="triage_request_id",
            max_chars=100,
        )
        text = st.text_area(
            "景区服务诉求",
            placeholder="例如：西门照明故障",
            max_chars=2000,
            height=140,
        )
        submitted = st.form_submit_button("提交分诊")

    if not submitted:
        return

    if not request_id.strip() or not text.strip():
        # 页面提前提示，API 仍保留对空字段转人工复核的最终保障。
        st.error("案件编号和景区服务诉求均不能为空。")
        return

    with st.spinner("正在调用本地 Agent..."):
        try:
            result = submit_triage_request(
                api_base_url=api_base_url,
                request_id=request_id,
                text=text,
            )
        except TriageApiClientError as error:
            st.error(str(error))
            return

    _render_case_result(result)


if __name__ == "__main__":
    main()
