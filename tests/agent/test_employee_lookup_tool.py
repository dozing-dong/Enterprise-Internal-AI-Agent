"""``employee_lookup`` 工具单测：通过 fake store 注入，避免真实 PG。

工具使用 ``InjectedToolCallId``，因此必须用 ToolCall 字典形式触发，
让框架自动注入 tool_call_id；返回值是 ``Command``。
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
    """触发 @tool 的标准 ToolCall 路径，由框架注入 tool_call_id。"""
    return tool.invoke(
        {
            "name": "employee_lookup",
            "args": args,
            "id": "test-call",
            "type": "tool_call",
        }
    )


def _command_observation(command: Command) -> dict:
    """从 ``Command.update.messages`` 取出工具回执 JSON。"""
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
    # store 收到的查询参数与签名一致。
    assert store.calls == [
        {"query": "alice", "department": None, "title": None, "limit": 5}
    ]
    # Command 还要把员工记录作为 sources 写回 state，
    # document_role 必须是 employee_structured，前端依赖该字段分区展示。
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
    # 没命中时不应该写 sources，避免污染前端 source 列表。
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
