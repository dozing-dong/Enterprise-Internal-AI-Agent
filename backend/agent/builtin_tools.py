"""Built-in agent tools.

- ``build_rag_answer_tool(rag_graph)``: wrap the compiled RAG LangGraph
  into a LangChain ``@tool``. When called, it synchronously invokes
  ``rag_graph.invoke(...)`` and uses ``Command`` to write the retrieved
  ``sources`` and ``retrieval_question`` back into the agent state, so the
  orchestrator can emit them in the final ``done`` event to the frontend.
- ``current_time``: a minimal tool demonstrating multi-tool routing.

session_id is injected from the agent graph state via ``InjectedState`` so
the tool signature stays clean (only parameters the LLM truly controls are exposed).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from backend.data.processing import convert_docs_to_sources
from backend.rag.employee_retriever import (
    EmployeeStore,
    employee_records_to_documents,
    safe_search_employees,
)


_RAG_ANSWER_DESCRIPTION = (
    "Answer a user question using the enterprise knowledge base. "
    "This tool runs the full retrieval-augmented-generation pipeline "
    "(query rewrite, hybrid retrieval, reranking, grounded generation) "
    "and returns a final answer along with the cited source snippets. "
    "Use this tool for any question that may require company knowledge, "
    "policy details, internal documents, or factual lookup. "
    "Prefer this tool over answering from your own memory."
)


def build_rag_answer_tool(rag_graph: Any):
    """Factory: bind the compiled RAG graph as a LangChain ``@tool``.

    Holds ``rag_graph`` via closure to avoid making it a global; also keeps
    the ``rag_answer`` tool signature down to a single ``question`` parameter,
    in line with the Bedrock Converse tool protocol.
    """

    @tool("rag_answer", description=_RAG_ANSWER_DESCRIPTION)
    def rag_answer(
        question: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """The user's question to answer using the knowledge base."""
        session_id = state.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                {"ok": False, "error": "missing session_id"},
                                ensure_ascii=False,
                            ),
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                    ]
                }
            )

        result = rag_graph.invoke(
            {"question": question, "session_id": session_id}
        )
        sources = result.get("sources", []) or []
        retrieval_question = result.get("retrieval_question", question)
        answer = result.get("answer", "") or ""

        observation = json.dumps(
            {
                "ok": True,
                "answer": answer,
                "retrieval_question": retrieval_question,
                "sources_count": len(sources),
            },
            ensure_ascii=False,
        )

        return Command(
            update={
                "sources": sources,
                "retrieval_question": retrieval_question,
                "messages": [
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call_id,
                        status="success",
                    )
                ],
            }
        )

    return rag_answer


_EMPLOYEE_LOOKUP_DESCRIPTION = (
    "Look up employees from the internal employee directory (PostgreSQL). "
    "Use this tool whenever the user asks who someone is, who works in a "
    "department, who holds a specific job title, or wants contact details. "
    "The tool performs a fuzzy match across name, department, title, "
    "employee_id and email. Optional filters `department` and `title` can "
    "further narrow the result. The tool returns a JSON object with `ok`, "
    "`results` (list of employee records) and `count`. An empty `results` "
    "list means no matching employee was found - in that case do NOT "
    "fabricate names, departments, or titles."
)


def build_employee_lookup_tool(store: EmployeeStore | None = None):
    """Factory: bind the employee store to the ``employee_lookup`` tool.

    Holds ``store`` via closure so tests can inject a fake; defaults to a
    new ``EmployeeStore`` sharing the production PG configuration.
    """
    bound_store = store if store is not None else EmployeeStore()

    @tool("employee_lookup", description=_EMPLOYEE_LOOKUP_DESCRIPTION)
    def employee_lookup(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        department: str | None = None,
        title: str | None = None,
        limit: int = 5,
    ) -> Command:
        """Search the employee directory for matching staff.

        Args:
            query: Free-text keyword. Matches name, department, title,
                employee_id and email (case-insensitive substring).
            department: Optional exact-ish department filter.
            title: Optional exact-ish job-title filter.
            limit: Maximum number of records to return (1-50).
        """
        records = safe_search_employees(
            bound_store,
            query,
            department=department,
            title=title,
            limit=limit,
        )

        observation = json.dumps(
            {
                "ok": True,
                "count": len(records),
                "results": [record.to_dict() for record in records],
            },
            ensure_ascii=False,
        )

        # Write employee records back into agent state in the same shape
        # as RAG sources. The frontend distinguishes "structured database
        # hit" sections by ``metadata.document_role == 'employee_structured'``.
        docs = employee_records_to_documents(records, query=query)
        structured_sources = convert_docs_to_sources(docs)

        update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call_id,
                    status="success",
                )
            ]
        }
        if structured_sources:
            update["sources"] = structured_sources

        return Command(update=update)

    return employee_lookup


@tool("current_time")
def current_time(timezone_name: str = "Pacific/Auckland") -> str:
    """Return the current date and time.

    Use this when the user explicitly asks what time it is, what today's
    date is, or needs the current timestamp for something. Do NOT use this
    for questions that are about knowledge in documents.

    Args:
        timezone_name: Optional IANA timezone name, e.g. 'Asia/Shanghai' or
            'UTC'. Defaults to Pacific/Auckland when omitted.
    """
    tz_name = timezone_name or "Pacific/Auckland"
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return json.dumps(
            {"ok": False, "error": f"unknown timezone: {tz_name}"},
            ensure_ascii=False,
        )

    now = datetime.now(tz)
    return json.dumps(
        {
            "ok": True,
            "iso": now.isoformat(),
            "timezone": str(tz),
            "epoch_seconds": int(now.timestamp()),
        },
        ensure_ascii=False,
    )
