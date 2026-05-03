"""PolicyAgent subgraph：内部差旅 / 报销 / 审批知识检索。

设计要点：
- 复用现有的 ``rag_answer`` 工具（由 ``backend.agent.builtin_tools``
  提供）。PolicyAgent 自己跑一个微型 ReAct 循环，确保模型真的去调
  RAG 而不是凭空生成。
- 子图运行结束后把 ``policy_result``（``answer`` / ``sources`` /
  ``retrieval_question``）写回顶层 ``MultiAgentState``，并追加
  ``"policy"`` 到 ``agents_invoked``。
- 子图内部使用一个**独立**的 messages channel（``inner_messages``）
  避免污染 supervisor / writer 看到的对话；只把工具产出的 sources
  合并到顶层 sources。
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
    """PolicyAgent 子图内部 state。

    使用 ``messages`` 作为 channel 名（而非 inner_messages）：
    rag_answer 工具的 ``Command(update={"messages": [...]})`` 是硬编码
    使用 ``messages`` 这个名字的，子图必须沿用以兼容。子图 state 与
    父图 state 是相互独立的命名空间，不会互相污染。
    """

    question: str
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    sources: Annotated[list[dict], _merge_unique_sources]
    retrieval_question: str | None
    answer: str


def build_policy_subgraph(rag_graph: Any):
    """构造 PolicyAgent 子图。``rag_graph`` 是已编译的 RAG LangGraph。"""

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
            # 保证 system prompt 始终在最前。
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(POLICY_SYSTEM_PROMPT), *messages]

        ai_msg = chat_model.invoke(messages)
        return {"messages": [ai_msg]}

    tool_node = ToolNode(tools, handle_tool_errors=lambda e: f"Tool error: {e}")

    def finalize(state: _PolicySubState) -> dict[str, Any]:
        """把内部对话压成 policy_result，并提取 final answer 字符串。"""
        answer_text = ""
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer_text = _flatten_text(msg.content)
                break

        retrieval_question = state.get("retrieval_question") or state.get("question") or ""
        return {"answer": answer_text, "retrieval_question": retrieval_question}

    def route_after_agent(state: _PolicySubState) -> str:
        """自定义条件：根据子图自己的 ``messages`` 判断是否还要调工具。"""
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
        """顶层节点：调用子图，返回写回顶层 state 的部分 dict。"""
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

        # 把内部 ReAct 的 tool 调用 + 结果抽成纯 dict，附在 policy_result
        # 里给 orchestrator 用，便于在顶层 trace 里渲染按 sub-agent 归属的
        # 工具调用条目。
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
    """从子图 messages 列表里抽出（call, result）配对成 dict。

    返回结构与 ExternalContextAgent 对齐：
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
    """把 ChatBedrockConverse 的 content（可能是 list）展平为字符串。"""
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
