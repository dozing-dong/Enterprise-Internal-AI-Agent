"""Agent decision layer.

Wraps the existing RAG pipeline into a LangChain ``@tool`` (``rag_answer``)
and uses LangGraph's ReAct loop to let the model autonomously choose tools
to produce an answer.
"""

from backend.agent.builtin_tools import (
    build_employee_lookup_tool,
    build_rag_answer_tool,
    current_time,
)
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
    "build_employee_lookup_tool",
    "build_initial_messages",
    "build_rag_answer_tool",
    "current_time",
]
