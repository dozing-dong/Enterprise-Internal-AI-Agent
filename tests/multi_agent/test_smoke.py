"""Multi-Agent 端到端 smoke test。

策略（minimal smoke，按计划 Q12-A）：
- monkeypatch ``backend.llm.chat_models.get_chat_model``，让 supervisor /
  policy / external / writer 各自看到的 LLM 都是 ``_FakeChatModel``，按
  脚本逐次返回 ``AIMessage``。Supervisor 的 ``with_structured_output(Plan)``
  路径返回一个固定 ``Plan``。
- 不真起 stdio MCP server；用一个本地 ``@tool`` 模拟天气查询，作为
  ``mcp_tools`` 注入 ExternalContextAgent。
- 用 ``MemoryHistoryStore`` 隔离历史副作用。
- 通过 ``ChatOrchestrator._stream_multi_agent`` 驱动一次完整对话，验证：
  1. supervisor / policy / external / writer 都进入 ``agents_invoked``，
     且按预期顺序。
  2. 仅 WriterAgent 的 token 出现在 ``token`` 事件里；policy / external
     的 LLM 输出不会被下发。
  3. ``done`` 事件 ``mode == "multi_agent"`` 且 ``agents_invoked`` 完整。
  4. ``trace`` 里同时存在 supervisor 节点 step 与 policy 子图工具调用 step。
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
    """脚本驱动的 fake ChatModel。

    - ``invoke`` / ``stream``：按脚本顺序产出 ``AIMessage``（含 / 不含
      ``tool_calls``）。
    - ``bind_tools`` / ``with_config``：返回自身，便于复用脚本。
    - ``with_structured_output(Plan)``：返回一个绑定了固定 ``Plan`` 的代理。
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
        # schema 必须是 Plan；返回另一个 model 代理，invoke 直接吐预设 Plan。
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
    """policy 子图里 ``rag_answer`` 工具背后的 fake RAG 图。"""

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
    """每个 sub-agent 各持一个独立 FakeChat。

    policy / external 在父图里是并行 fan-out 节点，共用一个 cursor 会出现
    竞态。给每个 sub-agent 自己的脚本就避免了这个问题。supervisor 不消耗
    cursor（走 structured-output 路径），脚本可为空。
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
    """驱动一次 ``_stream_multi_agent``，收集 SSE 事件。"""
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
    """完整的"出差 + 政策 + 天气 + 总结"路径。"""
    plan = Plan(
        use_policy=True,
        use_external=True,
        locations=["Auckland"],
        date_range="next week",
        needs_employee_lookup=False,
        rationale="travel question, needs both internal policy and weather",
    )

    # 每个 sub-agent 一份独立脚本：policy / external 在父图里并行 fan-out，
    # 共享一个 cursor 会有竞态。supervisor 走 structured-output 路径，
    # 脚本可空。
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

    # 1) 仅 writer 的内容出现在 done 的 full_answer 里：fake invoke 不会产
    #    生 messages 流的 chunk，但 orchestrator 会从 ``state.final_answer``
    #    兜底；policy / external 子图的中间产出不应进入最终答复。
    done = events[-1]
    full_answer = done.data["full_answer"]
    assert "Trip plan" in full_answer
    assert "Pre-approval" not in full_answer  # policy summary 不应泄露
    assert "Sunny" not in full_answer  # external summary 不应泄露

    # 真模型流式时，每个 token 事件都是 writer 的 chunk；fake 路径下允许为空。
    token_events = [e for e in events if e.type == "token"]
    joined = "".join(e.data["text"] for e in token_events)
    if joined:
        assert "Pre-approval" not in joined
        assert "Sunny" not in joined

    # 2) sources 由 PolicyAgent 写回（来自 fake rag_graph），通过 sources 事件
    #    暴露给前端。
    source_events = [e for e in events if e.type == "sources"]
    assert source_events, "expected at least one sources event"
    final_sources = source_events[-1].data["sources"]
    assert any(
        s.get("metadata", {}).get("context_id") == "policy_v1"
        for s in final_sources
    )

    # 3) done 事件包含完整的 agents_invoked 顺序。
    assert done.data["mode"] == "multi_agent"
    invoked = done.data["agents_invoked"]
    assert "supervisor" in invoked
    assert "policy" in invoked
    assert "external_context" in invoked
    assert "writer" in invoked
    # supervisor 必须排在 writer 前面。
    assert invoked.index("supervisor") < invoked.index("writer")

    # 4) trace 里同时有 supervisor / writer 的节点 step 与 policy 子图的
    #    rag_answer 工具调用 step。
    trace = done.data["trace"]
    names = {step["name"] for step in trace}
    assert "supervisor" in names
    assert "writer" in names
    assert any(
        step["name"] == "rag_answer" and step.get("agent") == "policy"
        for step in trace
    )

    # 5) Fake rag_graph 真的被 PolicyAgent 调用了一次。
    assert len(rag_graph.invoked_with) == 1


def test_multi_agent_skips_external_when_plan_disables_it(monkeypatch):
    """``Plan.use_external=False`` 时 external 子图不应被触发。"""
    plan = Plan(
        use_policy=True,
        use_external=False,
        locations=[],
        date_range=None,
        needs_employee_lookup=False,
        rationale="pure policy question",
    )

    policy_scripts = [
        # PolicyAgent: 直接给 final summary（不调用工具）。
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
    """``runtime.multi_agent_graph is None`` 时 stream 立即返回 error 事件。"""
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
    assert "Multi-Agent graph 未初始化" in events[-1].data["detail"]
