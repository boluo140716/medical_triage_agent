"""
LangGraph 业务节点：ReAct 思考节点 + 工具执行节点
合并原 think / judge / final_answer 三节点为双节点循环
"""
import contextvars
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from core.prompts import SYS_PROMPT
from tools.agent_tools import tool_list
from core.log_config import logger
from core.settings import LLM_MODEL_NAME, LLM_TEMPERATURE,MAX_TOOL_ROUNDS, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
# 知识库检索结果缓存（ContextVar 实现，多标签页/多请求隔离）
# LangGraph 会丢弃 AgentState TypedDict 中未声明的 key，因此使用 ContextVar 旁路缓存
_kb_docs_cache = contextvars.ContextVar("kb_docs_cache", default="")

# LLM 实例（懒加载，避免 import 时阻塞）
_llm = None


def _get_llm():
    """获取 LLM 实例（懒加载单例）"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            streaming=True,  # 启用流式回调，invoke() 时也能逐 token 推送
        )
    return _llm

# 上下文截断上限（字符数），约 2500~3000 tokens，避免消息膨胀拖慢 CPU 推理
MAX_CONTEXT_CHARS = 20000

# 假保存检测关键词（LLM 输出"复制保存"等文本但没调工具）
FAKE_SAVE_PATTERNS = ["复制保存", "复制以下", "请复制", "由于系统限制", "无法直接通过工具", "无法直接保存"]

# XML 工具调用标记（DeepSeek 可能输出文本形式 tool_calls，需从回答中剥离）
_XML_TOOL_MARKERS = ("<tool_call>", "<tool_calls>", "<｜tool_calls>", "<｜invoke", "</tool_call", "</｜tool_calls")


def _clean_xml_tool_content(content: str) -> str | None:
    """检测并剥离 XML 工具调用文本，返回纯文本；无有效内容返回 None"""
    if not any(tag in content for tag in _XML_TOOL_MARKERS):
        return None
    import re
    stripped = re.sub(r'<[｜tool_call|invoke|parameter|/].*?>', '', content, flags=re.DOTALL)
    stripped = re.sub(r'\n\s*\n', '\n\n', stripped).strip()
    return stripped if len(stripped) > 5 else ""


def _truncate_messages(messages: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """
    固定分段保留策略（不修改 state["docs"]）：
      系统提示词  →  永久保留
      用户提问(HumanMessage)  →  永久保留
      AI 历史对话(AIMessage)  →  永久保留
      检索结果(ToolMessage)  →  优先删减（新问题会重新检索，旧结果价值最低）

    保持时间顺序不变（不重排），从末尾逐条删除 ToolMessage 直到不超过上限。
    """
    if len(messages) <= 2:
        return messages

    # 计算总字符数
    total = sum(len(str(m.content)) for m in messages)
    if total <= max_chars:
        return messages

    # 从末尾逐条删除 ToolMessage
    result = list(messages)
    while len(result) > 1:
        current_total = sum(len(str(m.content)) for m in result)
        if current_total <= max_chars:
            break
        dropped = False
        for i in range(len(result) - 1, 0, -1):
            if isinstance(result[i], ToolMessage):
                result.pop(i)
                dropped = True
                break
        if not dropped:
            break

    # 最终兜底：若仍超限，等比压缩 AIMessage
    final_total = sum(len(str(m.content)) for m in result)
    if final_total > max_chars:
        ai_indices = [i for i, m in enumerate(result) if isinstance(m, AIMessage)]
        if ai_indices:
            tool_chars = sum(len(str(result[i].content)) for i in ai_indices)
            other_chars = final_total - tool_chars
            budget = max(500, max_chars - other_chars)
            per_msg = max(200, budget // len(ai_indices))
            for i in ai_indices:
                content = str(result[i].content)
                if len(content) > per_msg:
                    half = per_msg // 2
                    result[i] = AIMessage(content=content[:half] + "\n...[内容过长已精简]...\n" + content[-half:])

    if len(result) < len(messages):
        logger.info(
            f"上下文截断: {len(messages)} → {len(result)} 条消息 (约 {sum(len(str(m.content)) for m in result)} 字符)"
        )
    return result


async def agent_think_node(state):
    """
    ReAct 思考节点：LLM 决定调用工具或直接输出最终答案。

    工具绑定策略：
    - 知识库已检索到有效文档 → 移除 search_knowledge_base，禁止重复检索浪费轮数
    - 工具调用未达上限 → 绑定工具，LLM 可自由选择调工具或直接回答
    - 已达上限 → 不绑定工具，强制 LLM 输出纯文本答案
    """
    docs_cache = _kb_docs_cache.get()

    # 回退：若 ContextVar 未命中（LangGraph 可能隔离节点上下文），
    # 从 state["messages"] 中提取已有的 search_knowledge_base 检索结果
    if not docs_cache:
        for i, msg in enumerate(state.get("messages", [])):
            if isinstance(msg, ToolMessage) and msg.content and len(msg.content.strip()) > 10:
                if not msg.content.startswith("[工具异常]") and not msg.content.startswith("[系统错误]"):
                    # 校验消息来自 search_knowledge_base（非 search_online 等）
                    # 向前查找对应的 AIMessage tool_call 确认工具名称
                    for j in range(i - 1, -1, -1):
                        prev = state["messages"][j]
                        if hasattr(prev, "tool_calls"):
                            for tc in prev.tool_calls:
                                if tc.get("id") == msg.tool_call_id and tc.get("name") == "search_knowledge_base":
                                    docs_cache = msg.content
                                    _kb_docs_cache.set(docs_cache)
                                    break
                            if docs_cache:
                                break
                    if docs_cache:
                        break

    # 注入缓存的知识库文档（不受消息裁剪影响），确保 LLM 始终可见
    if docs_cache:
        enhanced_prompt = (
            SYS_PROMPT
            + "\n\n[已检索到的知识库文档——必须严格基于以下内容回答，禁止输出工具调用引导文本]\n"
            + docs_cache
        )
        raw_messages = [HumanMessage(content=enhanced_prompt)] + state["messages"]
    else:
        raw_messages = [HumanMessage(content=SYS_PROMPT)] + state["messages"]

    # 消除系统提示词重复注入：多轮 ReAct + 多轮对话中只保留最新一份 SYS_PROMPT
    # 多轮对话场景：不同轮次可能检索到不同 KB 文档，保留最新的 enhanced_prompt 确保 LLM 看到最新上下文
    deduped = []
    last_sys_idx = -1
    for i, msg in enumerate(raw_messages):
        if isinstance(msg, HumanMessage) and str(msg.content).startswith(SYS_PROMPT[:50]):
            last_sys_idx = i

    for i, msg in enumerate(raw_messages):
        if isinstance(msg, HumanMessage) and str(msg.content).startswith(SYS_PROMPT[:50]):
            if i != last_sys_idx:
                continue  # 跳过旧版系统提示词
        deduped.append(msg)
    raw_messages = deduped

    messages = _truncate_messages(raw_messages)

    # 统计已执行工具轮数（只算一次，后面复用）
    tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    force_answer = tool_count >= MAX_TOOL_ROUNDS

    available_tools = list(tool_list)
    llm = _get_llm()

    if force_answer:
        logger.info(f"强制纯文本回答模式 (工具轮数 {tool_count}/{MAX_TOOL_ROUNDS})")
        ai_msg = await llm.ainvoke(messages)
    else:
        llm_with_tools = llm.bind_tools(available_tools)
        ai_msg = await llm_with_tools.ainvoke(messages)

        if docs_cache and not ai_msg.content and not (hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls):
            logger.info("工具绑定模式下 KB 问答输出为空，回退到纯文本模式")
            ai_msg = await llm.ainvoke(messages)

    # 清理 XML 工具调用文本
    if ai_msg.content:
        cleaned = _clean_xml_tool_content(str(ai_msg.content))
        if cleaned:
            logger.info("检测到伪 tool_call XML 文本，已剥离")
            ai_msg = AIMessage(content=cleaned)
        elif cleaned is not None:
            logger.info("伪 tool_call XML 文本无有效内容，重试纯文本回答")
            ai_msg = await llm.ainvoke(messages)

    # 假保存检测
    if not force_answer:
        content = getattr(ai_msg, "content", "") or ""
        has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls
        if not has_tool_calls and any(p in content for p in FAKE_SAVE_PATTERNS):
            if tool_count >= MAX_TOOL_ROUNDS:
                logger.warning(f"检测到假保存文本但工具已达上限 {MAX_TOOL_ROUNDS}，放弃重试")
            else:
                logger.warning("检测到 LLM 假装保存（未调工具），追加提醒并强制重试")
                messages.append(HumanMessage(
                    content="【系统强制提醒】你刚才没有调用 save_summary_to_txt 工具！请立即调用 save_summary_to_txt 工具，将完整总结内容作为 summary_text 参数传入。禁止在回答中直接输出文本。"
                ))
                llm_with_tools = llm.bind_tools(available_tools)
                ai_msg = llm_with_tools.invoke(messages)

    return {"messages": [ai_msg]}


def tool_execute_node(state):
    """
    工具执行节点：执行 LLM 指定的工具，返回 ToolMessage。
    异常与空结果均保留可读标记，方便 LLM 判断下一步动作。

    search_knowledge_base 返回的有效文档会缓存到 ContextVar，
    后续消息裁剪不影响文档可用性，LLM 每轮都能看到检索结果。
    """
    last_msg = state["messages"][-1]
    tool_msg_collect = []
    docs_cache = _kb_docs_cache.get()

    for call_info in last_msg.tool_calls:
        tool_name = call_info.get("name", "")
        try:
            target_tool = next(t for t in tool_list if t.name == tool_name)
            tool_result = target_tool.invoke(call_info["args"])
            content = str(tool_result) if tool_result is not None else ""
        except StopIteration:
            logger.error(f"未找到工具: {tool_name}")
            content = f"[系统错误] 未找到工具: {tool_name}"
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行异常: {e}", exc_info=True)
            content = f"[工具异常] {type(e).__name__}: {str(e)}"

        # 知识库检索结果缓存到 ContextVar，不受 LangGraph 状态裁剪影响
        # 仅缓存有效文档内容，异常/空结果不缓存，保留重试机会
        if tool_name == "search_knowledge_base" and content and not docs_cache:
            if not content.startswith("[工具异常]") and not content.startswith("[系统错误]"):
                _kb_docs_cache.set(content)
                logger.info(f"知识库检索结果已缓存至 ContextVar ({len(content)} 字符)")

        tool_msg_collect.append(ToolMessage(
            content=content,
            tool_call_id=call_info.get("id", "")
        ))

    return {"messages": tool_msg_collect}