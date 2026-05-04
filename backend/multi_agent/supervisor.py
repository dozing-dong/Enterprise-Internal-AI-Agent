"""SupervisorAgent node.

Responsibilities:
- Receive the user question + history, call the LLM once
  (``with_structured_output(Plan)``) to produce a routing decision.
- If ``Plan.needs_employee_lookup`` is true, immediately call
  ``safe_search_employees`` to look up the employee directory (no ReAct
  loop), and write the results back to ``employee_context``.
- Append ``"supervisor"`` to ``agents_invoked``.

The Supervisor is just a regular LangGraph node function, not a separate
subgraph. The downstream ``Pol`` and ``Ext`` subgraphs are triggered via
conditional edges.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.data.processing import convert_docs_to_sources
from backend.llm import get_chat_model
from backend.multi_agent.policy import (
    AGENT_NAME_SUPERVISOR,
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_TEMPERATURE,
    Plan,
)
from backend.multi_agent.state import MultiAgentState
from backend.rag.employee_retriever import (
    EmployeeStore,
    employee_records_to_documents,
    safe_search_employees,
)


logger = logging.getLogger(__name__)


def _serialize_history(history: list[dict]) -> str:
    """Compress session history into short text for use as supervisor decision context."""
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        lines.append(f"{role}: {content[:240]}")
    return "\n".join(lines) or "(no prior turns)"


def build_supervisor_node(
    *,
    employee_store: EmployeeStore | None,
    employee_top_k: int,
):
    """Factory: return the supervisor node function (closes over employee_store)."""

    chat_model = get_chat_model(temperature=SUPERVISOR_TEMPERATURE)
    plan_model = chat_model.with_structured_output(Plan)

    def supervisor_node(state: MultiAgentState) -> dict[str, Any]:
        question = state.get("question") or ""
        history_text = _serialize_history(state.get("history") or [])

        prompt = [
            SystemMessage(SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(
                "Recent conversation:\n"
                f"{history_text}\n\n"
                f"User question: {question}\n\n"
                "Produce the Plan."
            ),
        ]

        try:
            plan: Plan = plan_model.invoke(prompt)
        except Exception:  # noqa: BLE001
            logger.exception(
                "supervisor structured-output failed, falling back to default plan"
            )
            plan = Plan(
                use_policy=True,
                use_external=False,
                rationale="fallback: supervisor LLM failed",
            )

        update: dict[str, Any] = {
            "plan": plan,
            "agents_invoked": [AGENT_NAME_SUPERVISOR],
        }

        if plan.needs_employee_lookup and employee_store is not None:
            try:
                records = safe_search_employees(
                    employee_store,
                    question,
                    limit=employee_top_k,
                )
            except Exception:  # noqa: BLE001
                logger.exception("supervisor employee_lookup failed")
                records = []
            if records:
                docs = employee_records_to_documents(records, query=question)
                structured_sources = convert_docs_to_sources(docs)
                update["employee_context"] = [r.to_dict() for r in records]
                if structured_sources:
                    update["sources"] = structured_sources

        return update

    return supervisor_node
