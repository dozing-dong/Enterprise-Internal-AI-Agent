"""Agent 决策层。

把现有 RAG 流水线封装为一个 LangChain ``@tool`` (``rag_answer``)，
通过 LangGraph 的 ReAct 循环让模型自主选择工具完成回答。
"""

from backend.agent.builtin_tools import build_rag_answer_tool, current_time
from backend.agent.graph import (
    AGENT_MAIN_TAG,
    AgentState,
    build_agent_graph,
    build_initial_messages,
)
from backend.agent.runner import AgentRunner

__all__ = [
    "AGENT_MAIN_TAG",
    "AgentRunner",
    "AgentState",
    "build_agent_graph",
    "build_initial_messages",
    "build_rag_answer_tool",
    "current_time",
]
