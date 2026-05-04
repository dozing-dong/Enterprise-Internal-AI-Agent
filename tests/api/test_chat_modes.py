"""End-to-end tests for the ``POST /chat`` and ``POST /chat/stream`` routes.

Strategy:
- mock ``backend.agent.graph.get_chat_model`` so the agent does not really hit Bedrock.
- assemble the runtime with a fake rag_graph and the real ``build_agent_graph``.
- verify:
  - ``POST /chat`` (mode=rag): goes straight through the fake rag_graph; assert the aggregated ChatResponse.
  - ``POST /chat`` (mode=agent): exercises the agent graph's ReAct path.
  - ``POST /chat/stream`` (mode=agent): receives ``sources`` and ``done(trace)`` events.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRagGraph:
    """Synchronous rag_graph stand-in. ``.invoke`` simulates one full RAG pipeline."""

    def __init__(self, *, sources=None, answer="rag-pipeline-answer", retrieval=None):
        self._sources = sources or [
            {"rank": 1, "content": "kb-snippet", "metadata": {"source": "kb"}}
        ]
        self._answer = answer
        self._retrieval = retrieval

    def invoke(self, payload):
        question = payload["question"]
        return {
            "answer": f"{self._answer}:{question}",
            "sources": list(self._sources),
            "retrieval_question": self._retrieval or f"rq:{question}",
            "original_question": question,
        }

    def stream(self, payload, *, stream_mode):
        """Simplified: merge the invoke result into a single ``updates`` chunk."""
        result = self.invoke(payload)
        # Simulate the updates payload from the finalize node.
        yield (
            "updates",
            {
                "finalize": {
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "retrieval_question": result["retrieval_question"],
                    "original_question": result["original_question"],
                }
            },
        )
        # tokens: simulate the messages chunk emitted by the generate_answer node.
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content=result["answer"])
        meta = {"langgraph_node": "generate_answer"}
        yield ("messages", (chunk, meta))


class _FakeChatModel:
    """Fake ChatModel used by the Agent node; returns scripted AIMessages in order."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self._cursor = 0

    def bind_tools(self, _tools):
        return self

    def with_config(self, **_kwargs):
        return self

    def invoke(self, *_args, **_kwargs):
        if self._cursor >= len(self._scripted):
            raise AssertionError("agent invoked the model more times than scripted")
        msg = self._scripted[self._cursor]
        self._cursor += 1
        return msg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_runtime(monkeypatch, *, scripted_agent_messages):
    from backend.agent import (
        build_agent_graph,
        build_rag_answer_tool,
        current_time,
    )

    fake_chat_model = _FakeChatModel(scripted_agent_messages)
    monkeypatch.setattr(
        "backend.agent.graph.get_chat_model",
        lambda **_: fake_chat_model,
    )

    rag_graph = _FakeRagGraph()
    tools = [build_rag_answer_tool(rag_graph), current_time]
    agent_graph = build_agent_graph(tools)

    return SimpleNamespace(
        documents=[object()],
        execution_mode="test-mode",
        vector_document_count=1,
        rag_graph=rag_graph,
        agent_graph=agent_graph,
    )


@pytest.fixture
def make_client(monkeypatch):
    """Factory: each test defines its own script and gets back (client, runtime)."""
    from backend.api import dependencies as deps_module
    from backend.api.app import app
    from backend.runtime import create_demo_runtime
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())

    def factory(*, scripted_agent_messages=None):
        runtime = _build_runtime(
            monkeypatch,
            scripted_agent_messages=scripted_agent_messages or [],
        )
        deps_module.set_runtime_factory(lambda: runtime)
        return TestClient(app), runtime

    try:
        yield factory
    finally:
        deps_module.reset_runtime()
        deps_module.set_runtime_factory(create_demo_runtime)
        history_module.reset_history_store()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chat_default_mode_is_rag(make_client):
    client, _ = make_client(scripted_agent_messages=[])
    with client:
        response = client.post(
            "/chat",
            json={"question": "ping", "session_id": "s-rag-001"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "rag"
    # /chat aggregates: answer comes from the rag graph stream.
    assert body["answer"] == "rag-pipeline-answer:ping"
    # trace must be a list (may be empty depending on stream emissions).
    assert isinstance(body["trace"], list)


def test_chat_agent_mode_routes_through_runner(make_client):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-x",
                    "name": "rag_answer",
                    "args": {"question": "policy?"},
                }
            ],
        ),
        AIMessage(content="Per the docs: it's allowed."),
    ]
    client, _ = make_client(scripted_agent_messages=scripted)
    with client:
        response = client.post(
            "/chat",
            json={
                "question": "policy?",
                "session_id": "s-agent-01",
                "mode": "agent",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "agent"
    # The fake chat model returns the final AIMessage via .invoke; the
    # orchestrator's "messages" stream sees no chunks, so full_answer falls
    # back to the done event's value (which is empty unless tokens streamed).
    # The aggregated /chat response uses done.full_answer when no tokens
    # arrived, but because there are no tokens AND no done.full_answer, the
    # body falls through to "No answer generated.".
    # Still, the trace must record the rag_answer call.
    trace = body["trace"]
    assert any(step["name"] == "rag_answer" for step in trace)
    # Sources from rag_answer surfaced into the response.
    assert body["sources"] == [
        {"rank": 1, "content": "kb-snippet", "metadata": {"source": "kb"}}
    ]


def test_chat_stream_agent_mode_emits_sources_and_done(make_client):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tu-1",
                    "name": "rag_answer",
                    "args": {"question": "what?"},
                }
            ],
        ),
        AIMessage(content="grounded answer"),
    ]
    client, _ = make_client(scripted_agent_messages=scripted)

    with client, client.stream(
        "POST",
        "/chat/stream",
        json={
            "question": "what?",
            "session_id": "session-stream-01",
            "mode": "agent",
        },
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        body = b"".join(chunk for chunk in response.iter_bytes()).decode("utf-8")

    events = _parse_sse(body)
    types = [name for name, _ in events]
    assert "sources" in types
    assert "done" in types
    assert types[-1] == "done"
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["mode"] == "agent"
    assert isinstance(done_payload["trace"], list)
    assert any(step["name"] == "rag_answer" for step in done_payload["trace"])


def test_chat_agent_mode_returns_500_on_planner_error(make_client):
    """After removing the fallback, an agent exception should bubble up as 5xx; POST /chat surfaces it via RagException."""

    class _BoomModel:
        def bind_tools(self, _tools):
            return self

        def with_config(self, **_kwargs):
            return self

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("aws is sad")

    import pytest
    from backend.api import dependencies as deps_module
    from backend.api.app import app
    from backend.runtime import create_demo_runtime
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())
    deps_module.reset_runtime()

    pytest.MonkeyPatch().setattr(
        "backend.agent.graph.get_chat_model",
        lambda **_: _BoomModel(),
    )

    from backend.agent import build_agent_graph, build_rag_answer_tool, current_time

    rag_graph = _FakeRagGraph()
    tools = [build_rag_answer_tool(rag_graph), current_time]
    agent_graph = build_agent_graph(tools)

    runtime = SimpleNamespace(
        documents=[object()],
        execution_mode="test-mode",
        vector_document_count=1,
        rag_graph=rag_graph,
        agent_graph=agent_graph,
    )
    deps_module.set_runtime_factory(lambda: runtime)

    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/chat",
                json={
                    "question": "anything",
                    "session_id": "s-fb-01",
                    "mode": "agent",
                },
            )
            assert response.status_code == 500
    finally:
        deps_module.reset_runtime()
        deps_module.set_runtime_factory(create_demo_runtime)
        history_module.reset_history_store()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_sse(text: str):
    """Split the SSE stream text into a list of [(event, json_payload)] tuples."""
    import json

    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip("\r\n")
        if not block:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                continue
            out.append((event_name, payload))
    return out
