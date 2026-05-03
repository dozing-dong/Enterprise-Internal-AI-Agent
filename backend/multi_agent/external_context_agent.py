"""ExternalContextAgent subgraph：通过 MCP 工具拉外部信息。

设计要点：
- 工具集来自 ``backend.mcp.load_external_mcp_tools()``：weather / web_search /
  business_calendar 等。工具在子图构造期注入；运行期不再 reload。
- 子图依然是一个微型 ReAct（``agent`` ⇄ ``tools`` 循环），但只接 MCP 工具，
  不接 RAG / 员工目录。
- 没拿到任何 MCP 工具时（环境无 Node / API key 全缺）子图自动降级为
  "直接产出 'no external tools available' 的纯文本结果"，依然返回结构化
  ``external_result``，不阻塞 Writer。
- 通过 tag ``agent_external`` 标记 LLM 调用，便于 orchestrator 过滤。
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
    """ExternalContextAgent 子图内部 state。

    用 ``messages`` 作为 channel 名，与 PolicyAgent 子图保持一致：MCP 工具
    返回标准 ``ToolMessage``，由 ToolNode 默认 append 到 ``messages``，无需
    自定义 messages_key。
    """

    question: str
    locations: list[str]
    date_range: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    tool_calls: list[dict]


def build_external_subgraph(mcp_tools: list[Any] | None):
    """构造 ExternalContextAgent 子图。

    ``mcp_tools`` 为空 / None 时，返回的节点只输出降级文本，不调用 LLM。
    """

    tools = list(mcp_tools or [])

    if not tools:
        # 无外部工具 → 纯降级路径，直接组装一个空的 external_result。
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

    tool_node = ToolNode(tools)

    def finalize(state: _ExternalSubState) -> dict[str, Any]:
        """把内部 messages 压成 external_result。"""
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
        """自定义条件：根据子图自己的 ``messages`` 判断是否还要调工具。"""
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
