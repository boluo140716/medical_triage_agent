"""
Gradio Web 界面：支持流式对话 + 临时文档上传 + 会话隔离
兼容 Gradio 6.x，对话历史使用元组格式 [(user, assistant), ...]
"""
import os
import traceback
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from settings import (
    load_dotenv,
    UPLOAD_MAX_FILE_SIZE_MB,
    UPLOAD_MAX_FILE_COUNT,
)
from agent.graph_builder import build_agent_graph
from document.loader import load_documents
from document.splitter import detail_splitter
from document.vector_store import embeddings
from log_config import logger
import session_store

# 加载环境变量、初始化智能体
load_dotenv()
agent_app = build_agent_graph()


def _extract_answer(result):
    """从 graph 最终状态提取可读答案，防御 tool_call 残留"""
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls and not last_msg.content:
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return msg.content
        return "抱歉，未能生成有效回答，请重试。"
    return last_msg.content or "（空回答）"


# ===================== 同步函数：上传 / 清空 =====================

def handle_upload(files, session_state):
    """
    处理用户上传文件：校验 → 加载 → 分片 → 创建内存 Chroma
    文件不落盘，仅存于当前会话内存 Chroma
    """
    if session_state is None:
        session_state = {"chroma": None, "file_names": [], "file_summaries": []}

    if files is None:
        return session_state, "⚠️ 未选择文件，请先选择文档后上传"

    # 统一为列表
    if not isinstance(files, list):
        files = [files]

    # 数量限制
    if len(files) > UPLOAD_MAX_FILE_COUNT:
        return session_state, f"❌ 单次最多上传 {UPLOAD_MAX_FILE_COUNT} 个文件，当前选择了 {len(files)} 个"

    existing_names = set(session_state.get("file_names", []))
    new_files = []       # (original_name, temp_path) 有效的新文件
    skip_names = []      # 重名跳过的文件名
    over_size_names = [] # 超大的文件名
    status_lines = []

    # 第一轮：逐个校验
    for f in files:
        try:
            # Gradio 6.x gr.File 返回格式兼容：dict 或 str
            if isinstance(f, dict):
                file_name = f.get("name", os.path.basename(f.get("path", "")))
                file_path = f.get("path", "")
            elif isinstance(f, str):
                file_name = os.path.basename(f)
                file_path = f
            else:
                logger.warning(f"未知文件格式: {type(f)}")
                continue

            if not file_path or not os.path.exists(file_path):
                logger.warning(f"文件路径无效: {file_path}")
                continue

            # 大小校验
            file_size_bytes = os.path.getsize(file_path)
            if file_size_bytes > UPLOAD_MAX_FILE_SIZE_MB * 1024 * 1024:
                over_size_names.append(file_name)
                continue

            # 重名校验
            if file_name in existing_names:
                skip_names.append(file_name)
                continue

            new_files.append((file_name, file_path))
            existing_names.add(file_name)

        except Exception as e:
            logger.error(f"文件校验异常: {e}", exc_info=True)
            continue

    # 校验反馈
    if over_size_names:
        status_lines.append(
            f"❌ 以下文件超过 {UPLOAD_MAX_FILE_SIZE_MB}MB 限制，已跳过：{', '.join(over_size_names)}"
        )
    if skip_names:
        status_lines.append(f"⚠️ 以下文件已存在，已跳过：{', '.join(skip_names)}")

    if not new_files:
        if not status_lines:
            status_lines.append("⚠️ 没有有效的文件需要上传")
        return session_state, "\n".join(status_lines)

    # 第二轮：加载文档并分片
    all_chunks = []
    success_files = []
    fail_files = []
    file_summaries = list(session_state.get("file_summaries", []))

    for file_name, file_path in new_files:
        try:
            docs = load_documents([file_path])
            if not docs:
                fail_files.append(file_name)
                continue

            # 生成摘要（取前 120 字符）
            full_text = docs[0].page_content if docs else ""
            summary = full_text[:120].replace("\n", " ").strip()
            if len(full_text) > 120:
                summary += "..."

            # 精细分片
            for doc in docs:
                chunks = detail_splitter.split_text(doc.page_content)
                all_chunks.extend(chunks)

            success_files.append(file_name)
            file_summaries.append({"name": file_name, "summary": summary})
            logger.info(f"上传文件解析成功: {file_name}, 分片数: {len(all_chunks) if docs else 0}")

        except Exception as e:
            logger.error(f"文件 {file_name} 解析失败: {e}", exc_info=True)
            fail_files.append(file_name)

    if fail_files:
        status_lines.append(f"❌ 以下文件解析失败：{', '.join(fail_files)}")

    if not all_chunks:
        status_lines.append("❌ 所有文件解析后均无有效内容")
        return session_state, "\n".join(status_lines)

    # 第三轮：创建/更新内存 Chroma
    try:
        # 获取已有 Chroma（若有），在其基础上追加
        existing_chroma = session_state.get("chroma")
        all_texts = []
        if existing_chroma is not None:
            try:
                # 从已有 Chroma 提取 texts
                existing_data = existing_chroma._collection.get()
                existing_docs = existing_data.get("documents", [])
                all_texts.extend(existing_docs)
            except Exception as e:
                logger.warning(f"无法读取已有 Chroma 数据，将仅使用新文件: {e}")
        all_texts.extend(all_chunks)

        new_chroma = Chroma.from_texts(
            texts=all_texts,
            embedding=embeddings,
        )
    except Exception as e:
        logger.error(f"创建内存 Chroma 失败: {e}", exc_info=True)
        return session_state, f"❌ 创建临时向量库失败：{str(e)}"

    # 更新会话状态
    session_state["chroma"] = new_chroma
    session_state["file_names"] = list(existing_names)
    session_state["file_summaries"] = file_summaries

    # 构建反馈信息
    status_lines.append(f"✅ 已索引 {len(success_files)} 个文件，共 {len(all_chunks)} 个分片")
    for fi in file_summaries:
        if fi["name"] in [s[0] for s in new_files]:
            status_lines.append(f"📄 **{fi['name']}** — {fi['summary']}")

    return session_state, "\n".join(status_lines)


def clear_upload(session_state):
    """仅清空上传的临时文件，保留对话记录"""
    try:
        if session_state:
            session_state.clear()
        logger.info("临时文档已清空")
    except Exception as e:
        logger.error(f"清空临时文档异常: {e}", exc_info=True)
    return session_state, "📭 已清空全部临时文档"


def clear_all(session_state):
    """清空对话记录 + 临时上传文件 + 输入框"""
    try:
        if session_state:
            session_state.clear()
        logger.info("已清空全部对话与临时文档")
    except Exception as e:
        logger.error(f"清空全部状态异常: {e}", exc_info=True)
    # 返回空聊天记录、重置的 session_state、友好提示、以及空输入框
    return [], {"chroma": None, "file_names": [], "file_summaries": []}, "📭 已清空对话与临时文档", ""


# ===================== 异步生成器：流式对话 =====================

async def chat_respond(user_input, chat_history, session_state):
    """
    处理用户提问：流式返回 LLM 生成的回答。

    **全局异常兜底**：最外层 try-except 捕获 LLM / 检索 / 上下文 / ContextVar
    等所有异常，确保无论如何都 yield 解锁页面输入框。

    - concurrency_limit=1 保证单会话串行执行
    - ContextVar 注入在生成循环外，finally 块确保清理
    - 对话历史使用元组格式 [(user, assistant), ...]
    """
    # 防御：任何未被 yield 阻断的路径都通过 return_early 兜底
    return_early = False

    try:
        # ===================== 全局 try：捕获一切异常 =====================

        if session_state is None:
            session_state = {"chroma": None, "file_names": [], "file_summaries": []}

        if not user_input or not user_input.strip():
            chat_history.append((user_input, "⚠️ 请输入有效问题"))
            yield "", chat_history, session_state
            return_early = True
            return

        # ---- ContextVar 注入：在生成循环外，确保 finally 可清理 ----
        chroma = session_state.get("chroma")
        if chroma is not None:
            try:
                session_store.set_current(chroma, session_state.get("file_names", []))
            except Exception as ctx_err:
                logger.error(f"ContextVar 注入异常: {ctx_err}", exc_info=True)

        # 初始化助手消息占位（元组格式）
        chat_history.append((user_input, ""))
        streamed_once = False

        # ===================== 中层：流式事件迭代 =====================
        try:
            async for event in agent_app.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                version="v2"
            ):
                try:
                    kind = event.get("event")

                    # LLM token 级流式输出
                    if kind == "on_chat_model_stream":
                        chunk_data = event.get("data", {}).get("chunk")
                        if chunk_data and hasattr(chunk_data, "content"):
                            content = chunk_data.content
                            if content:
                                new_answer = chat_history[-1][1] + content
                                chat_history[-1] = (user_input, new_answer)
                                yield "", chat_history, session_state
                                streamed_once = True

                except Exception as chunk_err:
                    # 单个 chunk 异常不中断整体流
                    logger.warning(f"流式 chunk 异常，跳过: {chunk_err}")
                    continue

            # 回退：未捕获到任何 token → invoke 补充
            if not streamed_once or chat_history[-1][1] == "":
                try:
                    result = agent_app.invoke({
                        "messages": [HumanMessage(content=user_input)]
                    })
                    answer = _extract_answer(result)
                    chat_history[-1] = (user_input, answer)
                    yield "", chat_history, session_state
                    streamed_once = True
                except Exception as invoke_err:
                    logger.error(f"回退 invoke 异常: {invoke_err}", exc_info=True)
                    chat_history[-1] = (user_input, f"抱歉，系统处理超时，请缩短问题后重试。")
                    yield "", chat_history, session_state
                    streamed_once = True

        except Exception as stream_err:
            # 流式迭代器级错误：记录完整堆栈，替换为友好提示
            logger.error(
                f"流式生成异常:\n{type(stream_err).__name__}: {stream_err}\n{traceback.format_exc()}"
            )
            chat_history[-1] = (user_input, f"抱歉，回答生成中断，请重试或换个问法。")
            yield "", chat_history, session_state

        finally:
            streamed_once = True  # 标记为已处理，防止外层重复 yield

    except Exception as fatal_err:
        # ===================== 最外层兜底：捕获一切漏网异常 =====================
        logger.error(
            f"对话执行致命异常:\n{type(fatal_err).__name__}: {fatal_err}\n{traceback.format_exc()}"
        )
        # 如果 chat_history 中没有当前轮次的错误记录，追加一条
        try:
            chat_history.append((user_input if user_input else "（空输入）",
                                 f"抱歉，系统出现异常。请点击「清空全部对话」后重新提问。"))
        except Exception:
            chat_history = [(user_input if user_input else "（空输入）",
                             f"抱歉，系统出现异常。请点击「清空全部对话」后重新提问。")]
        yield "", chat_history, session_state
        return

    finally:
        # ===================== 强制释放：ContextVar + 兜底 yield =====================
        try:
            session_store.clear_current()
        except Exception:
            pass

        # 兜底：确保输入框一定解锁
        # 如果整个函数执行中一次 yield 都没发生，Gradio 会永久锁住输入框
        if not return_early:
            try:
                yield "", chat_history, session_state
            except GeneratorExit:
                pass


# ===================== 构建 Web 界面 =====================
with gr.Blocks(title="企业知识库RAG智能问答系统") as demo:
    # 会话私有状态：每个浏览器标签页独立
    session_state = gr.State({"chroma": None, "file_names": [], "file_summaries": []})

    gr.Markdown("""
    # 🏢 企业内部知识库智能问答Agent
    功能：内部文档检索、行业资讯联网查询、文档内容总结导出
    """)

    chat_box = gr.Chatbot(
        height=500,
        label="对话记录",
    )

    # 折叠上传面板
    with gr.Accordion("📁 上传本地文档（临时会话）", open=False):
        upload_file = gr.File(
            label="选择文件",
            file_types=[".pdf", ".docx", ".txt", ".md", ".xlsx", ".xls"],
            file_count="multiple",
        )
        with gr.Row():
            upload_btn = gr.Button("上传并索引", variant="secondary")
            clear_upload_btn = gr.Button("清空上传文件", variant="stop", size="sm")
        upload_status = gr.Markdown("📂 尚未上传临时文档")

    input_text = gr.Textbox(
        label="提问输入框",
        placeholder="请输入问题，例如：试用期离职需要提前多久告知？",
        lines=2,
    )
    with gr.Row():
        submit_btn = gr.Button("提交提问", variant="primary")
        clear_btn = gr.Button("清空全部对话")

    # ---- 事件绑定 ----
    # 上传 / 清空（同步函数，仅修改 session_state）
    upload_btn.click(
        fn=handle_upload,
        inputs=[upload_file, session_state],
        outputs=[session_state, upload_status],
    )

    clear_upload_btn.click(
        fn=clear_upload,
        inputs=[session_state],
        outputs=[session_state, upload_status],
    )

    # 对话（异步生成器，流式返回，concurrency_limit=1 串行执行）
    submit_btn.click(
        fn=chat_respond,
        inputs=[input_text, chat_box, session_state],
        outputs=[input_text, chat_box, session_state],
        concurrency_limit=1,
    )
    input_text.submit(
        fn=chat_respond,
        inputs=[input_text, chat_box, session_state],
        outputs=[input_text, chat_box, session_state],
        concurrency_limit=1,
    )

    # 清空全部：同时重置输入框
    clear_btn.click(
        fn=clear_all,
        inputs=[session_state],
        outputs=[chat_box, session_state, upload_status, input_text],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
