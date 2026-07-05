"""
对话路由：SSE 流式响应 + 非流式回退
"""
import json
import os
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

from api.models import ChatRequest, ChatResponse
from api.dependency import inject_session, cleanup_session
from agent.graph_builder import agent_app
from core.session_utils import _extract_answer
from core.settings import TEMP_SUMMARY_DIR
from core.log_config import logger
from core import session_store
from core.prompts import SAVE_KEYWORDS

router = APIRouter(prefix="/api", tags=["对话"])


def _is_save_request(user_input: str) -> bool:
    """检测是否需要保存/导出"""
    return any(kw in user_input for kw in SAVE_KEYWORDS)


# XML 工具调用标记（DeepSeek 可能输出文本形式 tool_calls）
_XML_TOOL_MARKERS = ("<｜tool_calls", "<｜tool_call", "<tool_calls>", "<tool_call>", "<｜invoke", "<invoke")


def _has_xml_tool_start(text: str) -> bool:
    """检测文本中是否包含 XML 工具调用开始标记"""
    return any(m in text for m in _XML_TOOL_MARKERS)


def _strip_xml_content(text: str) -> str:
    """提取 XML 工具调用之前的内容，只保留纯文本部分"""
    import re
    # 找到第一个 XML 标记的位置，截取之前的内容
    first_idx = min(
        (text.find(m) for m in _XML_TOOL_MARKERS if m in text),
        default=-1
    )
    if first_idx > 0:
        return text[:first_idx].strip()
    # 兜底：正则剥离全部 XML 标签
    cleaned = re.sub(r'<[｜tool_call|invoke|parameter|/].*?>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
    return cleaned


async def _cleanup_incomplete_state(session_id: str):
    """清理因暂停留下的不完整工具调用状态，避免下次请求 DeepSeek 400 错误"""
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = await agent_app.aget_state(config)
        if not state or not state.values:
            return
        messages = list(state.values.get("messages", []))
        if not messages:
            return
        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
            return
        # 最后一条是带 tool_calls 的 AIMessage，说明上次暂停时工具未执行完
        messages.pop()
        for tc in last_msg.tool_calls:
            messages.append(ToolMessage(
                content="[操作已取消]",
                tool_call_id=tc.get("id", ""),
                name=tc.get("name", "unknown")
            ))
        await agent_app.aupdate_state(config, {"messages": messages})
        logger.info(f"会话 {session_id[:8]}... 已清理不完整工具调用状态")
    except Exception as e:
        logger.warning(f"清理状态失败: {e}")


async def _stream_chat_events(user_input: str, session_id: str):
    """
    SSE 事件生成器，逐 token 流式推送。

    事件格式：
      data: {"type": "token", "content": "你"}
      data: {"type": "token", "content": "好"}
      ...
      data: {"type": "done", "session_id": "abc123"}
      data: {"type": "clean", "content": "..."}  (XML 工具调用被过滤后的清理内容)
    """
    try:
        # 清理上次暂停留下的不完整工具调用状态
        await _cleanup_incomplete_state(session_id)

        graph_config = {"configurable": {"thread_id": session_id}}
        cancel_event = session_store.get_cancel_event(session_id)
        accumulated = ""  # 累积本轮 LLM 输出，用于检测 XML 工具调用
        xml_detected = False
        async for msg, _metadata in agent_app.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=graph_config,
            stream_mode="messages"
        ):
            if cancel_event.is_set():
                logger.info(f"会话 {session_id[:8]}... 收到取消请求，中断流式输出")
                yield f"data: {json.dumps({'type': 'cancelled', 'message': '已停止生成'}, ensure_ascii=False)}\n\n"
                return

            if isinstance(msg, AIMessageChunk) and msg.content:
                content = msg.content
                if isinstance(content, str):
                    accumulated += content
                    if not xml_detected and _has_xml_tool_start(accumulated):
                        xml_detected = True
                        clean = _strip_xml_content(accumulated)
                        if clean:
                            yield f"data: {json.dumps({'type': 'clean', 'content': clean}, ensure_ascii=False)}\n\n"
                    if not xml_detected:
                        yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

        # 检测是否有保存内容
        download_url = None
        filepath = os.path.join(TEMP_SUMMARY_DIR, session_id, "summary.txt")
        if os.path.isfile(filepath):
            download_url = f"/api/download/{session_id}"

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'download_url': download_url}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"SSE 流式异常: {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': '回答生成中断，请重试'}, ensure_ascii=False)}\n\n"
    finally:
        session_store.clear_cancel_event(session_id)
        cleanup_session()


@router.get("/download/{session_id}")
async def download_file(session_id: str):
    """
    下载保存的总结文件。
    浏览器收到 Content-Disposition: attachment 后会弹出"另存为"对话框。
    从文件读取会话持久化存储的总结内容。
    """
    filepath = os.path.join(TEMP_SUMMARY_DIR, session_id, "summary.txt")
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="没有可下载的内容，请先执行保存操作")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=summary_{session_id}.txt"
        },
    )


@router.post("/chat/stop")
async def stop_chat(req: ChatRequest):
    """停止当前会话的流式生成"""
    session_id = req.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    session_store.set_cancel_event(session_id)
    return {"status": "ok", "message": "已发送停止信号"}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE 流式对话接口。

    客户端使用 EventSource 或 fetch + ReadableStream 接收：
    - 逐 token 实时推送，改善体感延迟
    - 连接断开自动重连
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    save_requested = _is_save_request(req.question)
    session_id, _ = inject_session(req.session_id, save_allowed=save_requested)

    return StreamingResponse(
        _stream_chat_events(req.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            "X-Session-Id": session_id,
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    非流式对话接口（一次性返回完整回答）。

    适用场景：批量调用、自动化脚本、不需要流式的前端
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    save_requested = _is_save_request(req.question)
    session_id, _ = inject_session(req.session_id, save_allowed=save_requested)

    try:
        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content=req.question)]},
            config={"configurable": {"thread_id": session_id}},
        )
        answer = _extract_answer(result)

        # 检测是否有保存内容，返回下载链接
        download_url = None
        filepath = os.path.join(TEMP_SUMMARY_DIR, session_id, "summary.txt")
        if os.path.isfile(filepath):
            download_url = f"/api/download/{session_id}"

        return ChatResponse(answer=answer, session_id=session_id, download_url=download_url)
    except Exception as e:
        logger.error(f"非流式对话异常: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"系统异常: {str(e)}")
    finally:
        cleanup_session()