"""ExternalContextAgent subgraph: fetch external info via MCP tools.

Design notes:
- The tool set comes from ``backend.mcp.load_external_mcp_tools()``:
  weather / web_search / business_calendar etc. Tools are injected at
  subgraph construction time and not reloaded at runtime.
- The subgraph is a tiny ReAct (``agent`` <-> ``tools`` loop) but only
  binds MCP tools-not RAG or the employee directory.
- When no MCP tools are obtained (no Node available, all API keys
  missing), the subgraph automatically degrades to "directly produce a
  plain-text 'no external tools available' result", still returns a
  structured ``external_result`` and does not block the Writer.
- LLM calls are tagged with ``agent_external`` so the orchestrator can filter them.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from backend.llm import get_chat_model
from backend.multi_agent.policy import (
    AGENT_NAME_EXTERNAL,
    EXTERNAL_SYSTEM_PROMPT,
    EXTERNAL_TEMPERATURE,
)
from backend.multi_agent.state import MultiAgentState


logger = logging.getLogger(__name__)


EXTERNAL_AGENT_TAG = "agent_external"


class _ExternalSubState(TypedDict, total=False):
    """ExternalContextAgent subgraph internal state.

    Uses ``messages`` as the channel name to match the PolicyAgent
    subgraph: MCP tools return standard ``ToolMessage`` objects, which
    ToolNode appends to ``messages`` by default-no custom messages_key needed.
    """

    question: str
    locations: list[str]
    date_range: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    tool_calls: list[dict]


def build_external_subgraph(mcp_tools: list[Any] | None):
    """Build the ExternalContextAgent subgraph.

    When ``mcp_tools`` is empty / None, the returned node just produces
    fallback text and does not call the LLM.
    """

    tools = list(mcp_tools or [])

    if not tools:
        # No external tools -> pure fallback path; assemble an empty external_result.
        def fallback_node(state: MultiAgentState) -> dict[str, Any]:
            return {
                "external_result": {
                    "ok": True,
                    "answer": "No external tools are available in this environment.",
                    "tool_calls": [],
                    "tools_used": [],
                },
                "agents_invoked": [AGENT_NAME_EXTERNAL],
            }

        return fallback_node

    chat_model = (
        get_chat_model(temperature=EXTERNAL_TEMPERATURE)
        .bind_tools(tools)
        .with_config(tags=[EXTERNAL_AGENT_TAG])
    )

    def agent_node(state: _ExternalSubState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            question = state.get("question") or ""
            locations = ", ".join(state.get("locations") or []) or "(unspecified)"
            date_range = state.get("date_range") or "(unspecified)"
            messages = [
                SystemMessage(EXTERNAL_SYSTEM_PROMPT),
                HumanMessage(
                    f"User question: {question}\n"
                    f"Relevant locations: {locations}\n"
                    f"Time range: {date_range}\n\n"
                    "Use the available tools to gather objective external context, "
                    "then summarise."
                ),
            ]
        elif not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(EXTERNAL_SYSTEM_PROMPT), *messages]

        ai_msg = chat_model.invoke(messages)
        return {"messages": [ai_msg]}

    tool_node = ToolNode(tools, handle_tool_errors=lambda e: f"Tool error: {e}")

    def finalize(state: _ExternalSubState) -> dict[str, Any]:
        """Compress the internal messages into an external_result."""
        answer_text = ""
        tool_calls: list[dict] = []
        by_id: dict[str, dict] = {}

        messages = state.get("messages") or []
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in getattr(msg, "tool_calls", None) or []:
                    entry = {
                        "name": call.get("name", ""),
                        "args": call.get("args") or {},
                        "id": call.get("id", ""),
                        "ok": True,
                    }
                    tool_calls.append(entry)
                    if entry["id"]:
                        by_id[entry["id"]] = entry
            elif isinstance(msg, ToolMessage):
                payload_obj = _safe_json(getattr(msg, "content", ""))
                tool_call_id = getattr(msg, "tool_call_id", "") or ""
                target = by_id.get(tool_call_id)
                if target is not None:
                    target["result"] = payload_obj
                    if (
                        isinstance(payload_obj, dict)
                        and payload_obj.get("ok") is False
                    ):
                        target["ok"] = False
                        target["error"] = str(payload_obj.get("error", ""))

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer_text = _flatten_text(msg.content)
                break

        return {"answer": answer_text, "tool_calls": tool_calls}

    def route_after_agent(state: _ExternalSubState) -> str:
        """Custom condition: decide whether to call tools based on the subgraph's own ``messages``."""
        messages = state.get("messages") or []
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
        return "finalize"

    inner = StateGraph(_ExternalSubState)
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

    def external_node(state: MultiAgentState) -> dict[str, Any]:
        plan = state.get("plan")
        locations: list[str] = []
        date_range: str | None = None
        if plan is not None:
            locations = list(getattr(plan, "locations", []) or [])
            date_range = getattr(plan, "date_range", None)

        sub_input = _ExternalSubState(  # type: ignore[typeddict-item]
            question=state.get("question") or "",
            locations=locations,
            date_range=date_range,
            messages=[],
            answer="",
            tool_calls=[],
        )
        try:
            result = inner_compiled.invoke(sub_input)
        except Exception:  # noqa: BLE001
            logger.exception("ExternalContextAgent subgraph failed")
            return {
                "external_result": {
                    "ok": False,
                    "answer": "",
                    "tool_calls": [],
                    "tools_used": [],
                    "error": "external_agent_failed",
                },
                "agents_invoked": [AGENT_NAME_EXTERNAL],
            }

        tool_calls = result.get("tool_calls") or []
        tools_used = sorted({c.get("name", "") for c in tool_calls if c.get("name")})
        return {
            "external_result": {
                "ok": True,
                "answer": result.get("answer") or "",
                "tool_calls": tool_calls,
                "tools_used": tools_used,
            },
            "agents_invoked": [AGENT_NAME_EXTERNAL],
        }

    return external_node


def _flatten_text(content: Any) -> str:
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
    return ""


def _safe_json(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
