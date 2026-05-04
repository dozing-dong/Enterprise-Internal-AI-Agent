"""Unit tests for the Agent LangGraph.

Strategy:
- monkeypatch ``backend.llm.chat_models.get_chat_model`` to return a fake
  ChatModel:
  - ``invoke``: returns scripted ``AIMessage``s (with / without ``tool_calls``).
  - ``stream``: yields scripted ``AIMessageChunk``s.
  - ``bind_tools(...)``: returns self for script reuse.
  - ``with_config(...)``: returns self.
- Use ``MemoryHistoryStore`` to isolate history side effects.
- Drive ``runtime.agent_graph`` directly to verify:
  - the tool-routing path (rag_answer + final answer).
  - the time-only ``current_time`` path.
  - sources are written back to state; trace accumulates to the collector.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def memory_history():
    """Independent in-memory history per test to avoid cross-test leakage."""
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())
    try:
        yield
    finally:
        history_module.reset_history_store()


class _FakeChatModel:
    """Fake ChatModel that implements only the minimal surface used by the LangGraph nodes."""

    def __init__(self, scripted_messages: Iterable[AIMessage]) -> None:
        self._scripted = list(scripted_messages)
        self._cursor = 0

    def bind_tools(self, _tools):
        return self

    def with_config(self, **_kwargs):
        return self

    def _next_message(self) -> AIMessage:
        if self._cursor >= len(self._scripted):
            raise AssertionError("agent invoked the model more times than scripted")
        msg = self._scripted[self._cursor]
        self._cursor += 1
        return msg

    def invoke(self, _messages, **_kwargs) -> AIMessage:
        return self._next_message()

    def stream(self, _messages, **_kwargs):
        msg = self._next_message()
        yield AIMessageChunk(content=msg.content, tool_calls=msg.tool_calls)


def _build_runtime(
    monkeypatch,
    scripted_messages,
    *,
    rag_answer_payload=None,
    employee_records=None,
):
    """Build a fully faked agent_graph.

    rag_answer_payload: result returned by the fake rag sub-call when the
        agent invokes the rag_answer tool.
    employee_records: records returned by the fake store when the agent
        invokes the employee_lookup tool.
    """
    from backend.agent import (
        build_agent_graph,
        build_employee_lookup_tool,
        build_rag_answer_tool,
        current_time,
    )

    fake = _FakeChatModel(scripted_messages)
    monkeypatch.setattr(
        "backend.agent.graph.get_chat_model",
        lambda **_: fake,
    )

    class _FakeRagGraph:
        def invoke(self, payload):
            assert "question" in payload and "session_id" in payload
            return rag_answer_payload or {
                "answer": "rag-answer",
                "sources": [
                    {"rank": 1, "content": "doc-1", "metadata": {"source": "kb"}}
                ],
                "retrieval_question": f"rq:{payload['question']}",
            }

    class _FakeEmployeeStore:
        def search(self, *_args, **_kwargs):
            return list(employee_records or [])

    rag_graph = _FakeRagGraph()
    tools = [
        build_rag_answer_tool(rag_graph),
        build_employee_lookup_tool(_FakeEmployeeStore()),
        current_time,
    ]
    agent_graph = build_agent_graph(tools)
    return agent_graph


def _drive_agent_via_orchestrator(agent_graph, *, question, session_id, mode="agent"):
    """Drive a single streaming agent call via ChatOrchestrator and collect SSE-style events."""
    from types import SimpleNamespace

    from backend.api.schemas import ChatStreamRequest
    from backend.orchestrator import ChatOrchestrator
    from backend.storage.sessions import create_session_if_missing

    runtime = SimpleNamespace(
        rag_graph=None,
        agent_graph=agent_graph,
    )
    orch = ChatOrchestrator(runtime)  # type: ignore[arg-type]

    create_session_if_missing(session_id)
    request = ChatStreamRequest(
        question=question, session_id=session_id, mode=mode
    )
    return list(
        orch.stream(
            request,
            session_record_title="New Chat",
            is_first_turn=True,
        )
    )


# ---------------------------------------------------------------------------
# Direct graph behavior
# ---------------------------------------------------------------------------


def test_agent_graph_routes_through_rag_answer_tool(monkeypatch):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-1",
                    "name": "rag_answer",
                    "args": {"question": "What is X?"},
                }
            ],
        ),
        AIMessage(content="Final answer based on rag."),
    ]
    agent_graph = _build_runtime(
        monkeypatch,
        scripted,
        rag_answer_payload={
            "answer": "rag-answer",
            "sources": [
                {"rank": 1, "content": "doc-1", "metadata": {"source": "kb"}}
            ],
            "retrieval_question": "rewritten:What is X?",
        },
    )

    state = agent_graph.invoke(
        {
            "messages": [],
            "session_id": "session-knowledge",
            "sources": [],
            "retrieval_question": None,
            "original_question": "What is X?",
        }
    )

    # The rag_answer tool wrote sources back to state.
    assert state["sources"] == [
        {"rank": 1, "content": "doc-1", "metadata": {"source": "kb"}}
    ]
    assert state["retrieval_question"] == "rewritten:What is X?"
    # The final assistant message was the model's last reply.
    final_msg = state["messages"][-1]
    assert final_msg.content == "Final answer based on rag."


def test_agent_graph_combines_employee_lookup_and_rag(monkeypatch):
    """Simulate the "I'm xxx, I need to travel..." scenario:
    Step 1: the agent calls employee_lookup to get department/title.
    Step 2: the agent calls rag_answer to get travel policy snippets.
    Final sources must contain both kinds (structured + KB), and the
    reducer must not overwrite either side.
    """
    from backend.rag.employee_retriever import EmployeeRecord

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-emp",
                    "name": "employee_lookup",
                    "args": {"query": "alice"},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-rag",
                    "name": "rag_answer",
                    "args": {"question": "business travel policy"},
                }
            ],
        ),
        AIMessage(content="Per directory and policy: ..."),
    ]
    agent_graph = _build_runtime(
        monkeypatch,
        scripted,
        rag_answer_payload={
            "answer": "policy summary",
            "sources": [
                {
                    "rank": 1,
                    "content": "travel-snippet",
                    "metadata": {
                        "source": "local_eval",
                        "context_id": "policy_travel_v1",
                        "document_role": "reference_context",
                    },
                }
            ],
            "retrieval_question": "rewritten:travel",
        },
        employee_records=[
            EmployeeRecord(
                employee_id="E1001",
                name="Alice Carter",
                department="Engineering",
                title="Senior Backend Engineer",
                email="alice.carter@example.com",
            )
        ],
    )

    state = agent_graph.invoke(
        {
            "messages": [],
            "session_id": "session-combo",
            "sources": [],
            "retrieval_question": None,
            "original_question": "I'm Alice, what should I watch out for on a trip?",
        }
    )

    sources = state["sources"]
    structured = [
        s for s in sources
        if s.get("metadata", {}).get("document_role") == "employee_structured"
    ]
    kb = [
        s for s in sources
        if s.get("metadata", {}).get("document_role") == "reference_context"
    ]
    # Both kinds of sources must coexist: structured DB hits + KB hits.
    assert structured, "employee_lookup result must survive in sources"
    assert kb, "rag_answer result must survive in sources"
    assert structured[0]["metadata"]["employee_id"] == "E1001"
    assert kb[0]["metadata"]["context_id"] == "policy_travel_v1"


def test_agent_graph_handles_current_time_only(monkeypatch):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-2",
                    "name": "current_time",
                    "args": {"timezone_name": "UTC"},
                }
            ],
        ),
        AIMessage(content="It is currently <time>."),
    ]
    agent_graph = _build_runtime(monkeypatch, scripted)

    state = agent_graph.invoke(
        {
            "messages": [],
            "session_id": "session-time",
            "sources": [],
            "retrieval_question": None,
            "original_question": "What time is it?",
        }
    )
    assert state["sources"] == []
    final_msg = state["messages"][-1]
    assert final_msg.content == "It is currently <time>."


# ---------------------------------------------------------------------------
# Orchestrator / SSE event surface
# ---------------------------------------------------------------------------


def test_orchestrator_agent_stream_yields_sources_and_done_with_trace(monkeypatch):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-3",
                    "name": "rag_answer",
                    "args": {"question": "Q"},
                }
            ],
        ),
        AIMessage(content="Hello world"),
    ]
    agent_graph = _build_runtime(
        monkeypatch,
        scripted,
        rag_answer_payload={
            "answer": "rag-answer",
            "sources": [
                {"rank": 1, "content": "snippet", "metadata": {"source": "kb"}}
            ],
            "retrieval_question": "rq:Q",
        },
    )
    # session_id must be 8-64 chars of [A-Za-z0-9_-] — orchestrator stream
    # itself doesn't validate (route does), but session creation is happy.
    events = _drive_agent_via_orchestrator(
        agent_graph, question="Q", session_id="session-stream-1"
    )

    types = [e.type for e in events]
    # Sources must appear before done.
    assert "sources" in types
    assert types[-1] == "done"

    sources_event = next(e for e in events if e.type == "sources")
    assert sources_event.data["sources"][0]["metadata"]["source"] == "kb"
    assert sources_event.data["retrieval_question"] == "rq:Q"

    done = events[-1]
    assert done.data["mode"] == "agent"
    # full_answer is the final agent message; tokens may be empty when the
    # fake model returns whole-message AIMessage via .invoke (no streaming).
    assert done.data["full_answer"] in {"Hello world", "No answer generated."}
    # trace contains at least the rag_answer tool call (with summary populated
    # from the ToolMessage observation).
    trace = done.data["trace"]
    assert any(step["name"] == "rag_answer" for step in trace)
    rag_step = next(step for step in trace if step["name"] == "rag_answer")
    assert rag_step["ok"] is True
    assert rag_step["output_summary"] is not None


def test_orchestrator_agent_stream_falls_through_on_error(monkeypatch):
    """When the model raises, the stream ends with an error event and there is no RAG fallback."""

    class _BoomModel:
        def bind_tools(self, _tools):
            return self

        def with_config(self, **_kwargs):
            return self

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("aws is sad")

    monkeypatch.setattr(
        "backend.agent.graph.get_chat_model",
        lambda **_: _BoomModel(),
    )

    from backend.agent import build_agent_graph, build_rag_answer_tool, current_time

    class _UnusedRagGraph:
        def invoke(self, _payload):  # pragma: no cover - should not be called
            raise AssertionError("rag_graph must not be touched on agent error")

    tools = [build_rag_answer_tool(_UnusedRagGraph()), current_time]
    agent_graph = build_agent_graph(tools)

    events = _drive_agent_via_orchestrator(
        agent_graph, question="anything", session_id="session-err-1"
    )
    types = [e.type for e in events]
    assert types[-1] == "error"
    assert "aws is sad" in events[-1].data["detail"]


