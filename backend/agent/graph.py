"""Agent LangGraph: standard ReAct loop (agent <-> tools), fully streaming.

Design notes:
- Node ``agent``: invokes ``ChatBedrockConverse.bind_tools(...)``. Under
  ``stream_mode=["messages","updates"]``, the streaming tokens produced
  inside ``invoke`` are automatically captured by the ``messages`` stream
  as ``AIMessageChunk`` objects.
- Node ``tools``: ``langgraph.prebuilt.ToolNode`` automatically routes
  to the matching ``@tool`` based on ``AIMessage.tool_calls`` and supports
  ``Command`` return values that write business fields like ``sources``
  back into the state.
- After tool execution, control returns to the ``agent`` node automatically;
  ``tools_condition`` ends the run when there are no more ``tool_calls``,
  using the model's current AIMessage as the final answer.
- The agent's internal LLM call is tagged with
  ``with_config(tags=["agent_main"])`` so the orchestrator's streaming
  filter forwards only the user-visible final response and avoids
  double-emitting the internal generation of RAG-as-tool sub-calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from backend.agent.policy import AGENT_PLANNER_TEMPERATURE, AGENT_SYSTEM_PROMPT
from backend.llm import get_chat_model


AGENT_MAIN_TAG = "agent_main"


def _source_key(item: Any) -> tuple:
    """Generate a deduplication key for a source item.

    Prefers ``metadata.context_id`` (written by both RAG and
    employee_lookup); falls back to a content + rank tuple to avoid
    relying on dict hashability.
    """
    if not isinstance(item, dict):
        return ("__non_dict__", id(item))
    metadata = item.get("metadata") or {}
    context_id = ""
    if isinstance(metadata, dict):
        context_id = str(metadata.get("context_id", ""))
    content = str(item.get("content", ""))[:120]
    rank = item.get("rank")
    return (context_id, content, rank)


def _merge_unique(left: list, right: list) -> list:
    """LangGraph reducer: append after deduplication via ``_source_key``.

    Design goals:
    - Keep results from multiple tools in the same turn (``rag_answer`` +
      ``employee_lookup``) instead of overwriting one another whenever
      ``right`` is non-empty.
    - Empty writes still do not clear existing sources.
    """
    if not right:
        return list(left or [])

    seen: set[tuple] = set()
    merged: list = []
    for source in (left or [], right):
        for item in source:
            key = _source_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _keep_latest(left: Any, right: Any) -> Any:
    """LangGraph reducer: overwrite the old value with the new one (None / empty allowed)."""
    return right if right is not None else left


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    sources: Annotated[list[dict], _merge_unique]
    retrieval_question: Annotated[str | None, _keep_latest]
    original_question: str


def build_agent_graph(
    tools: Sequence,
    *,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
    temperature: float = AGENT_PLANNER_TEMPERATURE,
):
    """Build and compile the agent ReAct LangGraph.

    ``tools`` must be a list of already-built LangChain Tools (including
    ``rag_answer``).
    """
    chat_model = (
        get_chat_model(temperature=temperature)
        .bind_tools(list(tools))
        .with_config(tags=[AGENT_MAIN_TAG])
    )

    def agent_node(state: AgentState) -> dict:
        messages = list(state.get("messages", []))
        prompt = [SystemMessage(system_prompt), *messages]
        ai_msg = chat_model.invoke(prompt)
        return {"messages": [ai_msg]}

    tool_node = ToolNode(list(tools))

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_initial_messages(
    history: Sequence[dict],
    question: str,
) -> list[BaseMessage]:
    """Convert history + the current user question into a list of LangChain BaseMessages."""
    messages: list[BaseMessage] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        if role == "assistant":
            messages.append(AIMessage(content))
        elif role == "user":
            messages.append(HumanMessage(content))
    messages.append(HumanMessage(question))
    return messages
