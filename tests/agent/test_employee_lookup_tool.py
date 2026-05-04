"""Unit tests for the ``employee_lookup`` tool.

A fake store is injected so we never touch a real Postgres database.
The tool uses ``InjectedToolCallId`` so it must be triggered via the
ToolCall dict form, letting the framework auto-inject ``tool_call_id``;
the return value is a ``Command``.
"""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from backend.agent.builtin_tools import build_employee_lookup_tool
from backend.rag.employee_retriever import EmployeeRecord


class _FakeStore:
    def __init__(self, records: list[EmployeeRecord]):
        self._records = records
        self.calls: list[dict] = []

    def search(self, query, *, department=None, title=None, limit=5):
        self.calls.append(
            {"query": query, "department": department, "title": title, "limit": limit}
        )
        return list(self._records)


def _invoke(tool, args: dict) -> Command:
    """Trigger the standard @tool ToolCall path; the framework injects ``tool_call_id``."""
    return tool.invoke(
        {
            "name": "employee_lookup",
            "args": args,
            "id": "test-call",
            "type": "tool_call",
        }
    )


def _command_observation(command: Command) -> dict:
    """Extract the tool-result JSON payload from ``Command.update.messages``."""
    update = command.update
    messages = update["messages"] if isinstance(update, dict) else []
    assert messages and isinstance(messages[0], ToolMessage)
    return json.loads(messages[0].content)


def test_employee_lookup_returns_records_from_store():
    store = _FakeStore(
        [
            EmployeeRecord(
                employee_id="E1",
                name="Alice Carter",
                department="Engineering",
                title="SWE",
                email="alice@example.com",
            )
        ]
    )
    tool = build_employee_lookup_tool(store)
    command = _invoke(tool, {"query": "alice"})
    result = _command_observation(command)

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["employee_id"] == "E1"
    # The store received the query arguments matching the tool signature.
    assert store.calls == [
        {"query": "alice", "department": None, "title": None, "limit": 5}
    ]
    # The Command must also write employee records back to state as sources;
    # document_role must be employee_structured so the frontend can partition
    # the source list correctly.
    sources = command.update["sources"]
    assert sources and sources[0]["metadata"]["document_role"] == "employee_structured"
    assert sources[0]["metadata"]["employee_id"] == "E1"


def test_employee_lookup_returns_empty_when_no_match():
    store = _FakeStore([])
    tool = build_employee_lookup_tool(store)
    command = _invoke(tool, {"query": "nobody", "department": "Engineering"})
    result = _command_observation(command)

    assert result == {"ok": True, "count": 0, "results": []}
    assert store.calls[0]["department"] == "Engineering"
    # When there are no matches, sources must not be written, to avoid
    # polluting the frontend source list.
    assert "sources" not in command.update


def test_employee_lookup_passes_through_filters_and_limit():
    store = _FakeStore([])
    tool = build_employee_lookup_tool(store)
    _invoke(
        tool,
        {"query": "x", "department": "Sales", "title": "Lead", "limit": 10},
    )
    assert store.calls == [
        {"query": "x", "department": "Sales", "title": "Lead", "limit": 10}
    ]
