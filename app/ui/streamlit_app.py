"""景区服务诉求分诊与处置 Agent 的 Streamlit 演示页面。"""

from __future__ import annotations

import base64
import os
from html import escape
from datetime import datetime

import streamlit as st

from app.case_history.models import (
    CaseHistoryRecord,
    CaseHistoryResponse,
    HumanReviewQueueResponse,
)
from app.schemas.models import ServiceCaseResult
from app.ui.api_client import (
    TriageApiClientError,
    fetch_case_history,
    fetch_human_review_queue,
    submit_triage_request,
)


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

    if result.image is not None:
        st.subheader("图片观察")
        st.json(
            {
                "图片元数据": result.image.model_dump(mode="json"),
                "结构化观察": (
                    result.vision_observation.model_dump(mode="json")
                    if result.vision_observation is not None
                    else None
                ),
            }
        )

    if result.multimodal_fusion is not None:
        # 融合结果只来自受控概念和已校验的视觉观察，不能把它解释成真实图片识别结论。
        st.subheader("图文融合判断")
        st.json(
            {
                "状态": result.multimodal_fusion.status.value,
                "文本可核对概念": result.multimodal_fusion.text_concepts,
                "图片可核对概念": result.multimodal_fusion.image_concepts,
                "冲突字段": [
                    field.value
                    for field in result.multimodal_fusion.conflict_fields
                ],
                "说明": result.multimodal_fusion.note,
                "是否为本地演示判断": (
                    result.multimodal_fusion.is_demo_assessment
                ),
            }
        )

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


def _build_history_rows(records: list[CaseHistoryRecord]) -> list[dict[str, str]]:
    """将严格案件模型转换为适合表格扫描的展示字段。"""

    rows: list[dict[str, str]] = []
    for record in records:
        result = record.result
        rows.append(
            {
                "记录时间": record.recorded_at.astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "案件编号": result.request_id,
                "事件类别": result.classification.event_type.value,
                "风险等级": result.risk.level.value,
                "人工复核": "待复核" if record.requires_human_review else "不需要",
                "复核原因": "、".join(
                    reason.value for reason in result.review.reasons
                )
                or "-",
            }
        )
    return rows


def _render_records(
    records: list[CaseHistoryRecord],
    *,
    empty_message: str,
) -> None:
    """显示本地记录表格，并允许展开查看已校验的完整结果 JSON。"""

    if not records:
        st.info(empty_message)
        return

    st.dataframe(
        _build_history_rows(records),
        hide_index=True,
        use_container_width=True,
    )
    for record in records:
        result = record.result
        with st.expander(
            f"{result.request_id} · {result.risk.level.value} · "
            f"{record.recorded_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
        ):
            st.json(record.model_dump(mode="json"))


def _render_case_history_tab(api_base_url: str) -> None:
    """从本地 API 读取并展示最近保存的案件历史。"""

    st.subheader("案件历史")
    try:
        response: CaseHistoryResponse = fetch_case_history(
            api_base_url=api_base_url
        )
    except TriageApiClientError as error:
        st.error(str(error))
        return

    if response.storage_error:
        st.error(f"本地案件历史不可读取：{response.storage_error}")
        return
    _render_records(response.records, empty_message="尚未保存本地案件记录。")


def _render_review_queue_tab(api_base_url: str) -> None:
    """从本地 API 读取并展示所有仍需人工复核的案件。"""

    st.subheader("待人工复核")
    try:
        response: HumanReviewQueueResponse = fetch_human_review_queue(
            api_base_url=api_base_url
        )
    except TriageApiClientError as error:
        st.error(str(error))
        return

    if response.storage_error:
        st.error(f"待人工复核列表不可读取：{response.storage_error}")
        return
    _render_records(
        response.records,
        empty_message="当前没有待人工复核的本地案件。",
    )


def main() -> None:
    """创建页面表单；页面只发起请求，不重复实现 Agent 判断逻辑。"""

    st.set_page_config(page_title="景区服务诉求分诊", layout="wide")
    st.header("景区服务诉求分诊")
    st.caption("多模态服务诉求分诊与处置 Agent · 文本 + 图片受控观察演示")

    # 仅在本次浏览器会话首次加载页面时生成编号，之后允许用户稳定地手动修改。
    if "triage_request_id" not in st.session_state:
        st.session_state.triage_request_id = (
            f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    with st.sidebar:
        st.subheader("连接设置")
        api_base_url = st.text_input("FastAPI 服务地址", value=DEFAULT_API_BASE_URL)

    triage_tab, history_tab, review_queue_tab = st.tabs(
        ["提交分诊", "案件历史", "待人工复核"]
    )

    with triage_tab:
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
            uploaded_image = st.file_uploader(
                "上传图片（可选）",
                type=["jpg", "jpeg", "png", "webp"],
                help="当前使用本地演示视觉模型，只记录结构化观察，不代表真实像素识别。",
            )
            if uploaded_image is not None:
                # 上传阶段先限制大小，避免把超大文件编码后才被 API 拒绝。
                image_bytes = uploaded_image.getvalue()
                if len(image_bytes) > 5 * 1024 * 1024:
                    st.error("图片大小不能超过 5 MiB。")
                    uploaded_image = None
                else:
                    st.image(image_bytes, caption="本次待提交图片", width=260)
            submitted = st.form_submit_button("提交分诊")

        if submitted:
            if not request_id.strip() or not text.strip():
                # 页面提前提示，API 仍保留对空字段转人工复核的最终保障。
                st.error("案件编号和景区服务诉求均不能为空。")
            else:
                with st.spinner("正在调用本地 Agent..."):
                    try:
                        encoded_image = None
                        image_media_type = None
                        image_filename = None
                        if uploaded_image is not None:
                            # 仅在提交时编码，服务端处理结束后不会保存这段 base64。
                            encoded_image = base64.b64encode(
                                uploaded_image.getvalue()
                            ).decode("ascii")
                            image_media_type = uploaded_image.type
                            image_filename = uploaded_image.name
                        result = submit_triage_request(
                            api_base_url=api_base_url,
                            request_id=request_id,
                            text=text,
                            image_base64=encoded_image,
                            image_media_type=image_media_type,
                            image_filename=image_filename,
                            # 本地视觉模型首次载入权重需要较长时间，避免页面在
                            # 合理完成前按默认 15 秒超时；文本请求也可安全使用该值。
                            timeout_seconds=120.0,
                        )
                    except TriageApiClientError as error:
                        st.error(str(error))
                    else:
                        _render_case_result(result)

    with history_tab:
        _render_case_history_tab(api_base_url)

    with review_queue_tab:
        _render_review_queue_tab(api_base_url)


if __name__ == "__main__":
    main()
