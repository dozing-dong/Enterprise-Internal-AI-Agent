"""LangGraph：非流式 Agent 计划 → 执行工具 → 再计划，直到最终答复或用尽步数。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.agent.planner import plan_step
from backend.agent.schemas import AgentStep, ToolCall, ToolResult
from backend.agent.steps import build_tool_result_message
from backend.agent.tools import ToolRegistry


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    messages: list[dict[str, Any]]
    decision_trace: Annotated[list[dict], operator.add]
    sources: list[dict]
    retrieval_question: str | None
    step_index: int
    last_response: dict[str, Any]
    stop_reason: str
    final_text: str


def build_agent_graph(
    registry: ToolRegistry,
    *,
    system_prompt: str,
    max_steps: int,
) -> Any:
    graph = StateGraph(AgentState)

    def plan(state: AgentState) -> dict[str, Any]:
        response = plan_step(
            state["messages"],
            registry=registry,
            system_prompt=system_prompt,
        )
        new_index = state.get("step_index", 0) + 1
        return {
            "last_response": response,
            "stop_reason": response.get("stop_reason", ""),
            "step_index": new_index,
        }

    def execute_tool(state: AgentState) -> dict[str, Any]:
        response = state["last_response"]
        session_id = state["session_id"]
        step_index = state["step_index"]
        messages = state["messages"]
        step_text = response.get("text", "")
        tool_uses = response.get("tool_uses", [])
        first_use = tool_uses[0]
        call = ToolCall(
            name=first_use["name"],
            arguments=first_use.get("input") or {},
            tool_use_id=first_use["tool_use_id"],
        )
        result: ToolResult = registry.execute(
            call,
            context={"session_id": session_id},
        )

        patch: dict[str, Any] = {
            "decision_trace": [
                AgentStep(
                    index=step_index,
                    thought=step_text or None,
                    tool_call=call,
                    tool_result=result,
                ).to_dict()
            ],
            "messages": [
                *messages,
                {"role": "assistant", "content": response.get("raw_content", [])},
                build_tool_result_message(result),
            ],
        }

        if result.ok and call.name == "rag_answer":
            new_sources = result.data.get("sources", []) or []
            prev_sources = state.get("sources", []) or []
            patch["sources"] = new_sources or prev_sources
            merged_rq = result.data.get("retrieval_question") or state.get(
                "retrieval_question"
            )
            if merged_rq is not None:
                patch["retrieval_question"] = merged_rq

        return patch

    def finalize(state: AgentState) -> dict[str, Any]:
        response = state.get("last_response", {})
        step_index = state.get("step_index", 0)
        stop_reason = response.get("stop_reason", "")
        tool_uses = response.get("tool_uses", [])
        exhausted = (
            step_index >= max_steps
            and stop_reason == "tool_use"
            and bool(tool_uses)
        )
        if exhausted:
            return {"final_text": ""}

        final_text = response.get("text", "") or ""
        return {
            "final_text": final_text,
            "decision_trace": [
                AgentStep(
                    index=step_index,
                    final_answer=final_text,
                ).to_dict()
            ],
        }

    def route_after_plan(state: AgentState) -> str:
        stop_reason = state.get("stop_reason", "")
        tool_uses = state.get("last_response", {}).get("tool_uses", [])
        if stop_reason == "tool_use" and tool_uses:
            return "execute_tool"
        return "finalize"

    def route_after_execute(state: AgentState) -> str:
        step_index = state.get("step_index", 0)
        if step_index >= max_steps:
            return "finalize"
        return "plan"

    graph.add_node("plan", plan)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"execute_tool": "execute_tool", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_after_execute,
        {"plan": "plan", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
