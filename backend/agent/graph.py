"""Agent LangGraph：标准 ReAct 循环（agent ⇄ tools），全流式。

设计要点：
- 节点 ``agent``：调用 ``ChatBedrockConverse.bind_tools(...)``。在
  ``stream_mode=["messages","updates"]`` 下，``invoke`` 内部的流式 token
  会自动以 ``AIMessageChunk`` 形式被 ``messages`` 流捕获。
- 节点 ``tools``：``langgraph.prebuilt.ToolNode``，自动按
  ``AIMessage.tool_calls`` 路由到对应的 ``@tool``，支持 ``Command`` 返回值
  把 ``sources`` 等业务字段写回 state。
- 工具执行结束自动回到 ``agent`` 节点；``tools_condition`` 在没有更多
  ``tool_calls`` 时直接结束，由模型本轮的 AIMessage 作为最终答案。
- agent 内部 LLM 调用用 ``with_config(tags=["agent_main"])`` 打 tag，
  便于 orchestrator 在流式过滤中只转发 "用户可见" 的最终答复，避免
  RAG-as-tool 子调用的内部生成被双重下发。
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
    """生成 source 项的去重键。

    优先使用 ``metadata.context_id``（RAG 与 employee_lookup 都会写入该字段），
    退化到正文 + rank 组合，避免 dict 不可哈希。
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
    """LangGraph reducer：按 ``_source_key`` 去重后追加。

    设计目的：
    - 多工具同轮次写入（``rag_answer`` + ``employee_lookup``）都能保留，
      不再因为 “right 非空就整体覆盖” 而互相吞掉对方的结果。
    - 空写入仍然不会清空已有 sources。
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
    """LangGraph reducer：用新值覆盖旧值（None / 空字符串也允许覆盖）。"""
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
    """构建并编译 Agent ReAct LangGraph。

    ``tools`` 必须是已经构造好的 LangChain Tool 列表（含 ``rag_answer`` 等）。
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
    """把历史 + 当前用户问题转成 LangChain BaseMessage 列表。"""
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
