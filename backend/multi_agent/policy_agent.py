"""PolicyAgent subgraph: internal travel / expense / approval knowledge retrieval.

Design notes:
- Reuses the existing ``rag_answer`` tool (provided by
  ``backend.agent.builtin_tools``). The PolicyAgent runs its own tiny
  ReAct loop to ensure the model actually calls RAG instead of
  fabricating content.
- After the subgraph completes, write ``policy_result`` (``answer`` /
  ``sources`` / ``retrieval_question``) back into the top-level
  ``MultiAgentState`` and append ``"policy"`` to ``agents_invoked``.
- The subgraph uses an **independent** internal messages channel
  (``inner_messages``) to avoid polluting the conversation seen by the
  supervisor / writer; only the sources produced by tools are merged
  into the top-level sources.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from backend.agent.builtin_tools import build_rag_answer_tool
from backend.llm import get_chat_model
from backend.multi_agent.policy import (
    AGENT_NAME_POLICY,
    POLICY_SYSTEM_PROMPT,
    POLICY_TEMPERATURE,
)
from backend.multi_agent.state import MultiAgentState, _merge_unique_sources


logger = logging.getLogger(__name__)


POLICY_AGENT_TAG = "agent_policy"


class _PolicySubState(TypedDict, total=False):
    """PolicyAgent subgraph internal state.

    Uses ``messages`` as the channel name (not inner_messages):
    the rag_answer tool's ``Command(update={"messages": [...]})`` hard-codes
    the ``messages`` channel name, so the subgraph must follow suit for
    compatibility. The subgraph state and the parent graph state are
    independent namespaces and do not pollute each other.
    """

    question: str
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    sources: Annotated[list[dict], _merge_unique_sources]
    retrieval_question: str | None
    answer: str


def build_policy_subgraph(rag_graph: Any):
    """Build the PolicyAgent subgraph. ``rag_graph`` is the compiled RAG LangGraph."""

    rag_answer_tool = build_rag_answer_tool(rag_graph)
    tools = [rag_answer_tool]

    chat_model = (
        get_chat_model(temperature=POLICY_TEMPERATURE)
        .bind_tools(tools)
        .with_config(tags=[POLICY_AGENT_TAG])
    )

    def agent_node(state: _PolicySubState) -> dict[str, Any]:
        question = state.get("question") or ""
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [
                SystemMessage(POLICY_SYSTEM_PROMPT),
                HumanMessage(
                    f"User question: {question}\n\n"
                    "Decide whether to call rag_answer, then summarise."
                ),
            ]
        else:
            # Ensure the system prompt always sits at the front.
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(POLICY_SYSTEM_PROMPT), *messages]

        ai_msg = chat_model.invoke(messages)
        return {"messages": [ai_msg]}

    tool_node = ToolNode(tools, handle_tool_errors=lambda e: f"Tool error: {e}")

    def finalize(state: _PolicySubState) -> dict[str, Any]:
        """Compress the internal conversation into policy_result and extract the final answer string."""
        answer_text = ""
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer_text = _flatten_text(msg.content)
                break

        retrieval_question = state.get("retrieval_question") or state.get("question") or ""
        return {"answer": answer_text, "retrieval_question": retrieval_question}

    def route_after_agent(state: _PolicySubState) -> str:
        """Custom condition: decide whether to call tools based on the subgraph's own ``messages``."""
        messages = state.get("messages") or []
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
        return "finalize"

    inner = StateGraph(_PolicySubState)
    inner.add_node("agent", agent_node)
    inner.add_node("tools", tool_node)
    inner.add_node("finalize", finalize)
    inner.set_entry_point("agent")
    inner.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    inner.add_edge("tools", "agent")
    inner.add_edge("finalize", END)
    inner_compiled = inner.compile()

    def policy_node(state: MultiAgentState) -> dict[str, Any]:
        """Top-level node: invoke the subgraph and return a partial dict to merge back into the top-level state."""
        sub_input = _PolicySubState(  # type: ignore[typeddict-item]
            question=state.get("question") or "",
            session_id=state.get("session_id") or "",
            messages=[],
            sources=[],
            retrieval_question=None,
            answer="",
        )
        try:
            result = inner_compiled.invoke(sub_input)
        except Exception:  # noqa: BLE001
            logger.exception("PolicyAgent subgraph failed")
            return {
                "policy_result": {
                    "ok": False,
                    "answer": "",
                    "sources": [],
                    "retrieval_question": state.get("question") or "",
                    "error": "policy_agent_failed",
                },
                "agents_invoked": [AGENT_NAME_POLICY],
            }

        sources = result.get("sources") or []
        answer_text = result.get("answer") or ""
        retrieval_q = result.get("retrieval_question") or state.get("question") or ""

        # Extract the internal ReAct's tool calls + results into pure dicts,
        # attached to policy_result for the orchestrator to render
        # sub-agent-attributed tool-call entries in the top-level trace.
        tool_calls = _extract_tool_calls_from_messages(result.get("messages") or [])

        update: dict[str, Any] = {
            "policy_result": {
                "ok": True,
                "answer": answer_text,
                "sources": sources,
                "retrieval_question": retrieval_q,
                "tool_calls": tool_calls,
            },
            "agents_invoked": [AGENT_NAME_POLICY],
        }
        if sources:
            update["sources"] = sources
        return update

    return policy_node


def _extract_tool_calls_from_messages(messages: list[Any]) -> list[dict]:
    """Pull (call, result) pairs out of the subgraph's messages list as dicts.

    The returned shape mirrors that of ExternalContextAgent:
    ``[{"name": ..., "args": ..., "id": ..., "ok": True, "result": ...}, ...]``
    """
    calls: list[dict] = []
    by_id: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", None) or []:
                entry = {
                    "name": call.get("name", ""),
                    "args": call.get("args") or {},
                    "id": call.get("id", ""),
                    "ok": True,
                }
                calls.append(entry)
                if entry["id"]:
                    by_id[entry["id"]] = entry
        elif isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            entry = by_id.get(tool_call_id)
            payload_obj: Any
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                stripped = content.strip()
                try:
                    payload_obj = json.loads(stripped) if stripped else None
                except json.JSONDecodeError:
                    payload_obj = stripped
            else:
                payload_obj = content
            if entry is not None:
                entry["result"] = payload_obj
                if isinstance(payload_obj, dict) and payload_obj.get("ok") is False:
                    entry["ok"] = False
                    entry["error"] = str(payload_obj.get("error", ""))
            else:
                calls.append(
                    {
                        "name": getattr(msg, "name", "") or "tool",
                        "args": {},
                        "id": tool_call_id,
                        "ok": True,
                        "result": payload_obj,
                    }
                )
    return calls


def _flatten_text(content: Any) -> str:
    """Flatten a ChatBedrockConverse content (which can be a list) into a string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)
