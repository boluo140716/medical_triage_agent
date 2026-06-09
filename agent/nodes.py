"""
LangGraph 业务节点：ReAct 思考节点 + 工具执行节点
合并原 think / judge / final_answer 三节点为双节点循环
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from settings import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_GPU_NUM, MAX_TOOL_ROUNDS
from prompts import SYS_PROMPT
from tools.agent_tools import tool_list
from log_config import logger

# 初始化 LLM 实例
# 注意：numa 是 Ollama 服务端配置（启动时 OLLAMA_NUMA=true），不是 chat API 参数
# 不要在 extra_kwargs 中传递，否则新版 ollama 客户端会拒绝
llm = ChatOllama(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    num_gpu=LLM_GPU_NUM,
)

# 上下文截断上限（字符数），约 1500~2000 tokens，避免消息膨胀拖慢 CPU 推理
MAX_CONTEXT_CHARS = 6000


def _truncate_messages(messages: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """
    自动截断消息列表：保留系统提示词 + 最近的用户/工具/回答消息。
    控制总上下文长度，防止多轮工具调用后消息无限膨胀拖慢推理。
    """
    if len(messages) <= 2:
        return messages

    sys_msg = messages[0]  # 始终保留系统提示词
    rest = messages[1:]
    total = len(str(sys_msg.content))
    kept = []

    # 从最新消息往前保留，直到超出上限
    for msg in reversed(rest):
        content = msg.content if hasattr(msg, "content") else str(msg)
        msg_len = len(str(content))
        if total + msg_len > max_chars and kept:
            break
        kept.insert(0, msg)
        total += msg_len

    result = [sys_msg] + kept
    if len(result) < len(messages):
        logger.info(f"上下文截断: {len(messages)} → {len(result)} 条消息 (约 {total} 字符)")
    return result


def agent_think_node(state):
    """
    ReAct 思考节点：LLM 决定调用工具或直接输出最终答案。

    工具绑定策略：
    - 工具调用未达上限 → 绑定工具，LLM 可自由选择调工具或直接回答
    - 已达上限 → 不绑定工具，强制 LLM 输出纯文本答案，避免死循环
    """
    raw_messages = [HumanMessage(content=SYS_PROMPT)] + state["messages"]
    messages = _truncate_messages(raw_messages)

    # 统计已执行工具轮数，判断是否需要强制文本回答
    tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    force_answer = tool_count >= MAX_TOOL_ROUNDS

    if force_answer:
        # 已达工具上限：不绑定工具，强制 LLM 给出最终回答
        logger.info(f"工具已达上限 {MAX_TOOL_ROUNDS} 轮，强制 LLM 输出文本答案")
        ai_msg = llm.invoke(messages)
    else:
        # 未达上限：绑定工具，LLM 自行决定调用工具或直接回答
        llm_with_tools = llm.bind_tools(tool_list)
        ai_msg = llm_with_tools.invoke(messages)

    return {"messages": [ai_msg]}


def tool_execute_node(state):
    """
    工具执行节点：执行 LLM 指定的工具，返回 ToolMessage。
    异常与空结果均保留可读标记，方便 LLM 判断下一步动作。
    """
    last_msg = state["messages"][-1]
    tool_msg_collect = []

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

        tool_msg_collect.append(ToolMessage(
            content=content,
            tool_call_id=call_info["id"]
        ))

    return {"messages": tool_msg_collect}
