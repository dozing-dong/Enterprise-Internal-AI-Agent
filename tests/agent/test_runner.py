"""AgentRunner 的单元测试。

策略：
- mock ``backend.rag.models.chat_completion_with_tools`` 与
  ``chat_completion_with_tools_stream``，整段绕过 AWS / Bedrock。
- mock 一个 ``chat_executor`` 模拟现有 RAG 流水线返回值。
- 用 ``MemoryHistoryStore`` 隔离历史副作用。

覆盖：
- 知识问答 -> rag_answer 工具 -> 最终答复
- 时间问答 -> current_time 工具 -> 最终答复
- planner 抛错 -> 自动 fallback 到 chat_executor
- 流式：事件序列符合 progress / tool_call / tool_result / sources / token / done
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def memory_history():
    """每个用例独立的内存历史，避免跨用例残留。"""
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())
    try:
        yield
    finally:
        history_module.reset_history_store()


def _make_runtime(monkeypatch, *, planner_responses, fallback_answer="fallback-answer"):
    """构造一个不依赖 AWS 的 AgentRunner。

    ``planner_responses`` 是按调用顺序返回的列表，每一项就是一次
    ``chat_completion_with_tools`` 的返回值。
    """
    from backend.agent.runner import AgentRunner
    from backend.agent.tools import ToolRegistry
    from backend.agent.builtin_tools import CurrentTimeTool, RagAnswerTool

    call_log: list[dict[str, Any]] = []

    def fake_chat_executor(question: str, session_id: str) -> dict[str, Any]:
        call_log.append({"executor": "rag", "question": question, "session_id": session_id})
        return {
            "answer": fallback_answer,
            "original_question": question,
            "retrieval_question": f"rewritten:{question}",
            "sources": [{"rank": 1, "content": "doc-1", "metadata": {"source": "kb"}}],
            "tool_trace": [],
        }

    registry = ToolRegistry()
    registry.register(RagAnswerTool(fake_chat_executor))
    registry.register(CurrentTimeTool())

    response_iter = iter(planner_responses)

    def fake_plan(*args, **kwargs):
        try:
            return next(response_iter)
        except StopIteration as exc:
            raise AssertionError("planner called more times than scripted") from exc

    monkeypatch.setattr(
        "backend.agent.planner.chat_completion_with_tools",
        fake_plan,
    )

    runner = AgentRunner(
        registry,
        chat_executor_fallback=fake_chat_executor,
        max_steps=4,
    )
    return runner, call_log


def test_agent_run_uses_rag_answer_for_knowledge_question(monkeypatch):
    """模拟模型先调用 rag_answer，再用结果给出最终答复。"""
    planner_responses = [
        {
            "stop_reason": "tool_use",
            "text": "",
            "tool_uses": [
                {
                    "tool_use_id": "tu-1",
                    "name": "rag_answer",
                    "input": {"question": "What is X?"},
                }
            ],
            "raw_content": [
                {
                    "toolUse": {
                        "toolUseId": "tu-1",
                        "name": "rag_answer",
                        "input": {"question": "What is X?"},
                    }
                }
            ],
        },
        {
            "stop_reason": "end_turn",
            "text": "Final answer based on rag.",
            "tool_uses": [],
            "raw_content": [{"text": "Final answer based on rag."}],
        },
    ]
    runner, call_log = _make_runtime(monkeypatch, planner_responses=planner_responses)

    result = runner.run("What is X?", "session-knowledge")

    assert result.fallback is False
    assert result.answer == "Final answer based on rag."
    assert result.retrieval_question == "rewritten:What is X?"
    assert result.sources == [
        {"rank": 1, "content": "doc-1", "metadata": {"source": "kb"}}
    ]
    # rag_answer tool invokes the chat_executor exactly once; fallback never fires.
    assert call_log == [
        {"executor": "rag", "question": "What is X?", "session_id": "session-knowledge"}
    ]
    # decision_trace records 2 steps: tool call + final answer.
    assert len(result.decision_trace) == 2
    assert result.decision_trace[0]["tool_call"]["name"] == "rag_answer"
    assert result.decision_trace[0]["tool_result"]["ok"] is True
    assert result.decision_trace[1]["final_answer"] == "Final answer based on rag."


def test_agent_run_uses_current_time_tool(monkeypatch):
    """非知识问题：模型调用 current_time，无需走 rag_answer。"""
    planner_responses = [
        {
            "stop_reason": "tool_use",
            "text": "",
            "tool_uses": [
                {
                    "tool_use_id": "tu-2",
                    "name": "current_time",
                    "input": {"timezone": "UTC"},
                }
            ],
            "raw_content": [
                {
                    "toolUse": {
                        "toolUseId": "tu-2",
                        "name": "current_time",
                        "input": {"timezone": "UTC"},
                    }
                }
            ],
        },
        {
            "stop_reason": "end_turn",
            "text": "It is currently <time>.",
            "tool_uses": [],
            "raw_content": [{"text": "It is currently <time>."}],
        },
    ]
    runner, call_log = _make_runtime(monkeypatch, planner_responses=planner_responses)

    result = runner.run("What time is it?", "session-time")

    assert result.fallback is False
    assert result.answer == "It is currently <time>."
    # current_time tool does not touch chat_executor.
    assert call_log == []
    # No sources for non-rag tool path.
    assert result.sources == []
    assert result.decision_trace[0]["tool_call"]["name"] == "current_time"
    assert result.decision_trace[0]["tool_result"]["ok"] is True


def test_agent_run_falls_back_when_planner_raises(monkeypatch):
    """planner 抛错 -> fallback 到 chat_executor，仍能给出答复。"""

    def raising_plan(*args, **kwargs):
        raise RuntimeError("bedrock down")

    monkeypatch.setattr(
        "backend.agent.planner.chat_completion_with_tools",
        raising_plan,
    )

    from backend.agent.builtin_tools import CurrentTimeTool, RagAnswerTool
    from backend.agent.runner import AgentRunner
    from backend.agent.tools import ToolRegistry

    fallback_calls: list[tuple[str, str]] = []

    def fake_chat_executor(question: str, session_id: str) -> dict[str, Any]:
        fallback_calls.append((question, session_id))
        return {
            "answer": "fallback ok",
            "original_question": question,
            "retrieval_question": question,
            "sources": [],
            "tool_trace": [],
        }

    registry = ToolRegistry()
    registry.register(RagAnswerTool(fake_chat_executor))
    registry.register(CurrentTimeTool())

    runner = AgentRunner(
        registry,
        chat_executor_fallback=fake_chat_executor,
        max_steps=4,
    )

    result = runner.run("hello", "session-fallback")

    assert result.fallback is True
    assert result.answer == "fallback ok"
    assert fallback_calls == [("hello", "session-fallback")]
    # Last decision step records the fallback reason.
    assert any(
        step.get("fallback") and step.get("reason", "").startswith("RuntimeError")
        for step in result.decision_trace
    )


def test_agent_run_stream_emits_expected_event_order(monkeypatch):
    """流式路径的事件顺序：progress -> tool_call -> tool_result -> sources -> token -> done。"""
    from backend.agent.builtin_tools import CurrentTimeTool, RagAnswerTool
    from backend.agent.runner import AgentRunner
    from backend.agent.tools import ToolRegistry

    def fake_chat_executor(question, session_id):
        return {
            "answer": "rag final",
            "original_question": question,
            "retrieval_question": f"rq:{question}",
            "sources": [
                {"rank": 1, "content": "snippet", "metadata": {"source": "kb"}}
            ],
            "tool_trace": [],
        }

    registry = ToolRegistry()
    registry.register(RagAnswerTool(fake_chat_executor))
    registry.register(CurrentTimeTool())

    # Two consecutive plan_step_stream calls:
    # 1) emit a tool_use to call rag_answer.
    # 2) emit two text deltas + end_turn.
    streams = [
        iter(
            [
                {
                    "type": "tool_use",
                    "tool_use_id": "tu-s",
                    "name": "rag_answer",
                    "input": {"question": "Q"},
                },
                {"type": "stop", "stop_reason": "tool_use"},
            ]
        ),
        iter(
            [
                {"type": "text_delta", "text": "Hello "},
                {"type": "text_delta", "text": "world"},
                {"type": "stop", "stop_reason": "end_turn"},
            ]
        ),
    ]

    def fake_plan_stream(*args, **kwargs):
        return next(streams_iter)

    streams_iter = iter(streams)

    monkeypatch.setattr(
        "backend.agent.planner.chat_completion_with_tools_stream",
        fake_plan_stream,
    )

    runner = AgentRunner(
        registry,
        chat_executor_fallback=fake_chat_executor,
        max_steps=4,
    )

    events = list(runner.run_stream("Q", "session-stream"))
    types = [event.type for event in events]

    # Loose ordering check: must contain these in this relative order.
    assert types[0] == "progress"  # initial deciding
    tool_call_idx = types.index("tool_call")
    tool_result_idx = types.index("tool_result")
    sources_idx = types.index("sources")
    first_token_idx = types.index("token")
    done_idx = types.index("done")

    assert tool_call_idx < tool_result_idx < sources_idx < first_token_idx < done_idx

    # Two text deltas surface as two `token` events.
    token_events = [event for event in events if event.type == "token"]
    assert [e.data["text"] for e in token_events] == ["Hello ", "world"]

    # done payload carries fallback=False and the streamed full_answer.
    # `mode` / `title` are appended by the route layer, not the runner.
    done_event = events[-1]
    assert done_event.type == "done"
    assert done_event.data["fallback"] is False
    assert done_event.data["full_answer"] == "Hello world"
    assert done_event.data["session_id"] == "session-stream"


def test_agent_run_max_steps_exhausted_falls_back(monkeypatch):
    """如果模型一直要求调工具，达到 max_steps 后兜底到 RAG。"""
    from backend.agent.builtin_tools import CurrentTimeTool, RagAnswerTool
    from backend.agent.runner import AgentRunner
    from backend.agent.tools import ToolRegistry

    fallback_calls: list[str] = []

    def fake_chat_executor(question, session_id):
        fallback_calls.append(question)
        return {
            "answer": "fallback after max_steps",
            "original_question": question,
            "retrieval_question": question,
            "sources": [],
            "tool_trace": [],
        }

    # Always return tool_use, never end_turn.
    looping_response = {
        "stop_reason": "tool_use",
        "text": "",
        "tool_uses": [
            {
                "tool_use_id": "tu-loop",
                "name": "current_time",
                "input": {},
            }
        ],
        "raw_content": [
            {
                "toolUse": {
                    "toolUseId": "tu-loop",
                    "name": "current_time",
                    "input": {},
                }
            }
        ],
    }

    monkeypatch.setattr(
        "backend.agent.planner.chat_completion_with_tools",
        lambda *args, **kwargs: looping_response,
    )

    registry = ToolRegistry()
    registry.register(RagAnswerTool(fake_chat_executor))
    registry.register(CurrentTimeTool())

    runner = AgentRunner(
        registry,
        chat_executor_fallback=fake_chat_executor,
        max_steps=2,
    )

    result = runner.run("loop", "session-loop")

    assert result.fallback is True
    assert result.answer == "fallback after max_steps"
    assert fallback_calls == ["loop"]
