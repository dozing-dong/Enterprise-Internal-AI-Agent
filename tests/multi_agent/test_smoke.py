"""Multi-Agent end-to-end smoke test.

Strategy (minimal smoke, plan Q12-A):
- monkeypatch ``backend.llm.chat_models.get_chat_model`` so that supervisor /
  policy / external / writer each see a ``_FakeChatModel`` and return
  ``AIMessage`` according to a per-agent script. Supervisor's
  ``with_structured_output(Plan)`` path returns a fixed ``Plan``.
- We do not start a real stdio MCP server; a local ``@tool`` simulates a
  weather lookup and is injected into ExternalContextAgent as ``mcp_tools``.
- ``MemoryHistoryStore`` isolates history side effects.
- Drive a full conversation through ``ChatOrchestrator._stream_multi_agent``
  and verify:
  1. supervisor / policy / external / writer all show up in
     ``agents_invoked`` in the expected order.
  2. Only WriterAgent tokens appear in ``token`` events; the
     policy / external LLM outputs are not forwarded.
  3. The ``done`` event has ``mode == "multi_agent"`` and a complete
     ``agents_invoked`` list.
  4. ``trace`` contains both supervisor node steps and the policy subgraph's
     tool-call steps.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from backend.multi_agent.policy import Plan


# ---------------------------------------------------------------------------
# Fixtures & fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def memory_history():
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())
    try:
        yield
    finally:
        history_module.reset_history_store()


class _FakeChatModel:
    """Script-driven fake ChatModel.

    - ``invoke`` / ``stream``: yield ``AIMessage`` (with / without
      ``tool_calls``) in script order.
    - ``bind_tools`` / ``with_config``: return self so the script can be reused.
    - ``with_structured_output(Plan)``: returns a proxy that ``invoke``s a
      preset ``Plan``.
    """

    def __init__(self, scripted_messages: Iterable[AIMessage]) -> None:
        self._scripted = list(scripted_messages)
        self._cursor = 0
        self._structured_plan: Plan | None = None

    # ---- LangGraph node interface -------------------------------------

    def bind_tools(self, _tools):
        return self

    def with_config(self, *_args, **_kwargs):
        return self

    def with_structured_output(self, schema):
        # The schema must be Plan; return a separate model proxy whose
        # ``invoke`` directly emits the preset Plan.
        owner = self

        class _StructuredProxy:
            def invoke(self, _messages, **_kwargs):
                if owner._structured_plan is None:
                    raise AssertionError(
                        "structured Plan was not configured on the fake model"
                    )
                return owner._structured_plan

        return _StructuredProxy()

    def set_structured_plan(self, plan: Plan) -> None:
        self._structured_plan = plan

    # ---- script driving ----------------------------------------------

    def _next(self) -> AIMessage:
        if self._cursor >= len(self._scripted):
            raise AssertionError(
                "fake model invoked more times than scripted: "
                f"cursor={self._cursor}, scripted={len(self._scripted)}"
            )
        msg = self._scripted[self._cursor]
        self._cursor += 1
        return msg

    def invoke(self, _messages, **_kwargs) -> AIMessage:
        return self._next()

    def stream(self, _messages, **_kwargs):
        msg = self._next()
        yield AIMessageChunk(content=msg.content, tool_calls=msg.tool_calls)


class _FakeRagGraph:
    """Fake RAG graph backing the ``rag_answer`` tool inside the policy subgraph."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.invoked_with: list[dict] = []

    def invoke(self, payload):
        self.invoked_with.append(payload)
        return self.payload


@tool("weather_lookup")
def fake_weather_lookup(city: str) -> str:
    """Return a fake weather report for the given city (used in tests)."""
    return f"sunny in {city}, 18-22 degC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_multi_agent_graph_with_fakes(
    monkeypatch,
    *,
    plan: Plan,
    supervisor_scripts: Iterable[AIMessage] | None = None,
    policy_scripts: Iterable[AIMessage] | None = None,
    external_scripts: Iterable[AIMessage] | None = None,
    writer_scripts: Iterable[AIMessage] | None = None,
):
    """Each sub-agent owns its own FakeChat instance.

    policy / external are parallel fan-out nodes in the parent graph;
    sharing a single cursor would cause races. Giving each sub-agent its
    own script avoids that. The supervisor does not consume the cursor
    (it goes through the structured-output path), so its script may be empty.
    """
    supervisor_fake = _FakeChatModel(supervisor_scripts or [])
    supervisor_fake.set_structured_plan(plan)
    policy_fake = _FakeChatModel(policy_scripts or [])
    external_fake = _FakeChatModel(external_scripts or [])
    writer_fake = _FakeChatModel(writer_scripts or [])

    monkeypatch.setattr(
        "backend.multi_agent.supervisor.get_chat_model",
        lambda **_: supervisor_fake,
    )
    monkeypatch.setattr(
        "backend.multi_agent.policy_agent.get_chat_model",
        lambda **_: policy_fake,
    )
    monkeypatch.setattr(
        "backend.multi_agent.external_context_agent.get_chat_model",
        lambda **_: external_fake,
    )
    monkeypatch.setattr(
        "backend.multi_agent.writer_agent.get_chat_model",
        lambda **_: writer_fake,
    )

    from backend.multi_agent.graph import build_multi_agent_graph

    rag_graph = _FakeRagGraph(
        payload={
            "answer": "Travel must be pre-approved by line manager.",
            "sources": [
                {
                    "rank": 1,
                    "content": "policy snippet",
                    "metadata": {
                        "source": "kb",
                        "context_id": "policy_v1",
                        "document_role": "reference_context",
                    },
                }
            ],
            "retrieval_question": "rewritten:travel",
        }
    )

    graph = build_multi_agent_graph(
        rag_graph=rag_graph,
        mcp_tools=[fake_weather_lookup],
        employee_store=None,
    )
    return graph, rag_graph


def _drive(graph, *, question: str, session_id: str):
    """Drive one ``_stream_multi_agent`` run and collect SSE events."""
    from backend.api.schemas import ChatStreamRequest
    from backend.orchestrator import ChatOrchestrator
    from backend.storage.sessions import create_session_if_missing

    runtime = SimpleNamespace(
        rag_graph=None,
        agent_graph=None,
        multi_agent_graph=graph,
    )
    orch = ChatOrchestrator(runtime)  # type: ignore[arg-type]
    create_session_if_missing(session_id)
    request = ChatStreamRequest(
        question=question, session_id=session_id, mode="multi_agent"
    )
    return list(
        orch.stream(
            request,
            session_record_title="New Chat",
            is_first_turn=True,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_agent_routes_through_policy_external_writer(monkeypatch):
    """The full "travel + policy + weather + summary" path."""
    plan = Plan(
        use_policy=True,
        use_external=True,
        locations=["Auckland"],
        date_range="next week",
        needs_employee_lookup=False,
        rationale="travel question, needs both internal policy and weather",
    )

    # One independent script per sub-agent: policy / external are parallel
    # fan-out nodes in the parent graph, so sharing a single cursor would
    # race. The supervisor goes through the structured-output path, so its
    # script may be empty.
    policy_scripts = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "pol-1",
                    "name": "rag_answer",
                    "args": {"question": "travel policy"},
                }
            ],
        ),
        AIMessage(
            content=(
                "Relevant rules:\n- Pre-approval needed.\n"
                "Unclear / missing:\n- Per-diem rate."
            )
        ),
    ]
    external_scripts = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "ext-1",
                    "name": "weather_lookup",
                    "args": {"city": "Auckland"},
                }
            ],
        ),
        AIMessage(
            content=(
                "Weather:\n- Sunny, 18-22 degC.\n"
                "Holidays / calendar:\n- (none).\n"
                "Web / news:\n- (skipped)."
            )
        ),
    ]
    writer_scripts = [
        AIMessage(content="### Trip plan\n1. Recap\n2. Approvals\n3. Weather"),
    ]

    graph, rag_graph = _build_multi_agent_graph_with_fakes(
        monkeypatch,
        plan=plan,
        policy_scripts=policy_scripts,
        external_scripts=external_scripts,
        writer_scripts=writer_scripts,
    )

    events = _drive(
        graph,
        question="I'm flying to Auckland next week. Help me plan it.",
        session_id="session-multi-1",
    )

    types = [e.type for e in events]
    assert types[-1] == "done"

    # 1) Only writer content appears in done.full_answer: the fake invoke
    #    does not produce a messages-stream chunk, but the orchestrator
    #    falls back to ``state.final_answer``; intermediate output from
    #    the policy / external subgraphs must not leak into the final answer.
    done = events[-1]
    full_answer = done.data["full_answer"]
    assert "Trip plan" in full_answer
    assert "Pre-approval" not in full_answer  # policy summary must not leak
    assert "Sunny" not in full_answer  # external summary must not leak

    # When the real model streams, every token event is a writer chunk;
    # under the fake path it is allowed to be empty.
    token_events = [e for e in events if e.type == "token"]
    joined = "".join(e.data["text"] for e in token_events)
    if joined:
        assert "Pre-approval" not in joined
        assert "Sunny" not in joined

    # 2) Sources are written back by the PolicyAgent (from the fake
    #    rag_graph) and exposed to the frontend through sources events.
    source_events = [e for e in events if e.type == "sources"]
    assert source_events, "expected at least one sources event"
    final_sources = source_events[-1].data["sources"]
    assert any(
        s.get("metadata", {}).get("context_id") == "policy_v1"
        for s in final_sources
    )

    # 3) The done event contains the full agents_invoked sequence.
    assert done.data["mode"] == "multi_agent"
    invoked = done.data["agents_invoked"]
    assert "supervisor" in invoked
    assert "policy" in invoked
    assert "external_context" in invoked
    assert "writer" in invoked
    # Supervisor must come before writer.
    assert invoked.index("supervisor") < invoked.index("writer")

    # 4) The trace contains both supervisor / writer node steps and the
    #    policy subgraph's rag_answer tool-call step.
    trace = done.data["trace"]
    names = {step["name"] for step in trace}
    assert "supervisor" in names
    assert "writer" in names
    assert any(
        step["name"] == "rag_answer" and step.get("agent") == "policy"
        for step in trace
    )

    # 5) The fake rag_graph really was invoked once by the PolicyAgent.
    assert len(rag_graph.invoked_with) == 1


def test_multi_agent_skips_external_when_plan_disables_it(monkeypatch):
    """When ``Plan.use_external=False`` the external subgraph must not be triggered."""
    plan = Plan(
        use_policy=True,
        use_external=False,
        locations=[],
        date_range=None,
        needs_employee_lookup=False,
        rationale="pure policy question",
    )

    policy_scripts = [
        # PolicyAgent: emit a final summary directly (no tool call).
        AIMessage(content="Relevant rules:\n- Pre-approval needed."),
    ]
    writer_scripts = [
        AIMessage(content="Final policy-only answer."),
    ]

    graph, _rag_graph = _build_multi_agent_graph_with_fakes(
        monkeypatch,
        plan=plan,
        policy_scripts=policy_scripts,
        writer_scripts=writer_scripts,
    )

    events = _drive(
        graph,
        question="What is the travel approval rule?",
        session_id="session-multi-2",
    )
    done = events[-1]
    invoked = done.data["agents_invoked"]
    assert "supervisor" in invoked
    assert "policy" in invoked
    assert "writer" in invoked
    assert "external_context" not in invoked


def test_multi_agent_orchestrator_returns_503_when_graph_missing(monkeypatch):
    """When ``runtime.multi_agent_graph is None`` the stream should immediately emit an error event."""
    runtime = SimpleNamespace(
        rag_graph=None,
        agent_graph=None,
        multi_agent_graph=None,
    )
    from backend.api.schemas import ChatStreamRequest
    from backend.orchestrator import ChatOrchestrator
    from backend.storage.sessions import create_session_if_missing

    create_session_if_missing("session-multi-3")
    orch = ChatOrchestrator(runtime)  # type: ignore[arg-type]
    events = list(
        orch.stream(
            ChatStreamRequest(
                question="anything",
                session_id="session-multi-3",
                mode="multi_agent",
            ),
            session_record_title="New Chat",
            is_first_turn=True,
        )
    )
    assert events[-1].type == "error"
    assert "Multi-Agent graph is not initialized" in events[-1].data["detail"]
