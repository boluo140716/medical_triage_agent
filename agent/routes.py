"""
路由判断函数：控制 ReAct 图的分支走向
"""
from langgraph.graph import END
from langchain_core.messages import ToolMessage
from core.settings import MAX_TOOL_ROUNDS
from core.log_config import logger
from agent.nodes import FAKE_SAVE_PATTERNS


def tool_route_func(state) -> str:
    """
    判断 think 节点输出后的路由：
    - 无 tool_calls → END（LLM 已输出最终文本答案）
    - 有 tool_calls 且未到上限 → tool_execute_node
    - 有 tool_calls 且已达上限 → agent_think_node（不绑工具，强制文本回答）
    - 检测到假保存文本 → agent_think_node（强制重试）
    """
    last_msg = state["messages"][-1]

    # 无工具调用 → LLM 已给出最终答案
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        content = getattr(last_msg, "content", "") or ""
        if any(p in content for p in FAKE_SAVE_PATTERNS):
            logger.warning("检测到 LLM 假装保存（未调工具），强制返回 think 重试")
            return "agent_think_node"
        logger.info("LLM 输出最终文本答案，结束")
        return END

    # 统计已执行的工具消息数
    tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))

    # 未达工具上限 → 执行工具
    if tool_count < MAX_TOOL_ROUNDS:
        return "tool_execute_node"

    # 已达上限 → 回到 think 节点，但不绑工具，强制输出文本答案
    logger.info(f"工具已达上限 {MAX_TOOL_ROUNDS} 轮，路由到强制回答模式")
    return "agent_think_node"