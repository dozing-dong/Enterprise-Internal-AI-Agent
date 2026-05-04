"""Multi-Agent shared state.

Design notes:
- ``MultiAgentState`` defines the channels of the top-level LangGraph;
  the Supervisor and the three subgraphs share this single state, with
  subgraphs writing back fields like ``policy_result`` / ``external_result``
  via returned dicts.
- To support Policy / External fan-out, ``policy_result`` and
  ``external_result`` use the ``_keep_latest`` reducer; ``sources`` and
  ``agents_invoked`` use a deduplicating merge reducer.
- This module **does not** depend on ``backend.agent``, avoiding a cycle
  between the single- and multi-agent packages.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from backend.multi_agent.policy import Plan


def _source_key(item: Any) -> tuple:
    """Generate a deduplication key for a sources item (matches backend.agent.graph)."""
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
    """LangGraph reducer: append sources after deduplication via ``_source_key``."""
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
    """LangGraph reducer: ``agents_invoked`` appended in order with deduplication."""
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
    """LangGraph reducer: overwrite the old value with the new one; None is not treated as overwrite."""
    return right if right is not None else left


class MultiAgentState(TypedDict, total=False):
    """State shared by the top-level multi-agent graph.

    All sub-agents update these channels by returning a partial dict.
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
    """Build the initial state for a new conversation turn."""
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
