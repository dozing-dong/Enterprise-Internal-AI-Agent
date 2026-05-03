"""Multi-Agent 共享状态。

设计要点：
- ``MultiAgentState`` 是顶层 LangGraph 的 channels；Supervisor 与三个子图
  共享同一份 state，子图通过返回 dict 写回 ``policy_result`` /
  ``external_result`` 等字段。
- 为了支持 Policy / External 并行 fan-out，``policy_result`` 与
  ``external_result`` 通过 ``_keep_latest`` reducer 接收子图返回值；
  ``sources`` / ``agents_invoked`` 用合并去重 reducer。
- 该模块**不**依赖 ``backend.agent``，避免单 / 多 Agent 包之间形成循环。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from backend.multi_agent.policy import Plan


def _source_key(item: Any) -> tuple:
    """生成 sources 项的去重键（与 backend.agent.graph 形态一致）。"""
    if not isinstance(item, dict):
        return ("__non_dict__", id(item))
    metadata = item.get("metadata") or {}
    context_id = ""
    if isinstance(metadata, dict):
        context_id = str(metadata.get("context_id", ""))
    content = str(item.get("content", ""))[:120]
    rank = item.get("rank")
    return (context_id, content, rank)


def _merge_unique_sources(left: list, right: list) -> list:
    """LangGraph reducer：sources 按 ``_source_key`` 去重后追加。"""
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


def _append_unique_strings(left: list[str], right: list[str]) -> list[str]:
    """LangGraph reducer：``agents_invoked`` 按出现顺序追加，去重。"""
    if not right:
        return list(left or [])

    seen: set[str] = set(left or [])
    merged: list[str] = list(left or [])
    for item in right:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _keep_latest(left: Any, right: Any) -> Any:
    """LangGraph reducer：用新值覆盖旧值，None 不视为覆盖。"""
    return right if right is not None else left


class MultiAgentState(TypedDict, total=False):
    """顶层 multi-agent 图共享的状态。

    所有 sub-agent 都通过返回部分 dict 来更新这些 channel。
    """

    question: str
    session_id: str
    history: list[dict]

    plan: Annotated[Plan | None, _keep_latest]
    employee_context: Annotated[list[dict], _keep_latest]

    policy_result: Annotated[dict | None, _keep_latest]
    external_result: Annotated[dict | None, _keep_latest]

    sources: Annotated[list[dict], _merge_unique_sources]
    agents_invoked: Annotated[list[str], _append_unique_strings]

    final_answer: Annotated[str, _keep_latest]
    messages: Annotated[list[BaseMessage], add_messages]


def build_initial_multi_agent_state(
    *,
    question: str,
    session_id: str,
    history: list[dict] | None = None,
) -> MultiAgentState:
    """构造一份新对话回合的初始 state。"""
    return MultiAgentState(  # type: ignore[typeddict-item]
        question=question,
        session_id=session_id,
        history=list(history or []),
        plan=None,
        employee_context=[],
        policy_result=None,
        external_result=None,
        sources=[],
        agents_invoked=[],
        final_answer="",
        messages=[],
    )
