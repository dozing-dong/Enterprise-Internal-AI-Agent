"""内置 Agent 工具。

- ``build_rag_answer_tool(rag_graph)``：把编译好的 RAG LangGraph 包装成一个
  LangChain ``@tool``。该工具在被调用时同步 ``rag_graph.invoke(...)``，
  并通过 ``Command`` 把检索到的 ``sources`` 与 ``retrieval_question``
  写回 agent 状态，供 orchestrator 在最终 ``done`` 事件里下发给前端。
- ``current_time``：极简工具，演示多工具路由能力。

session_id 通过 ``InjectedState`` 从 agent 图状态注入，工具签名保持
干净（只暴露给 LLM 真正可控的参数）。
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
    """工厂：把已编译的 RAG 图绑定为一个 LangChain ``@tool``。

    通过闭包持有 ``rag_graph``，避免把 graph 写到全局；
    同时让 ``rag_answer`` 的工具签名只暴露 ``question``，符合
    Bedrock Converse 的工具协议。
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
    """工厂：把员工存储绑定为 ``employee_lookup`` 工具。

    通过闭包持有 ``store`` 引用，便于测试中注入 fake；默认创建一个
    ``EmployeeStore``，与生产 PG 共用配置。
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

        # 把员工记录以与 RAG sources 同构的形状写回 agent 状态。
        # 前端按 ``metadata.document_role == 'employee_structured'`` 区分
        # 出"结构化数据库命中"区块。
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
