"""POST /chat 路由 mode 开关的端到端测试。

测试模型：依赖注入一个真实 ``AgentRunner`` + 真实 ``ToolRegistry``，
但 mock 掉 Bedrock 调用与 RAG 流水线，避免外部依赖。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


def _fake_chat_executor(question: str, session_id: str) -> dict[str, Any]:
    return {
        "answer": f"rag-pipeline-answer:{question}",
        "original_question": question,
        "retrieval_question": f"rq:{question}",
        "sources": [
            {"rank": 1, "content": "kb-snippet", "metadata": {"source": "kb"}}
        ],
        "tool_trace": [],
    }


def _build_runtime():
    """构造真实 DemoRuntime 替身，agent_runner 使用真实工具但绕过 LLM。"""
    from types import SimpleNamespace

    from backend.agent.builtin_tools import CurrentTimeTool, RagAnswerTool
    from backend.agent.runner import AgentRunner
    from backend.agent.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(RagAnswerTool(_fake_chat_executor))
    registry.register(CurrentTimeTool())

    agent_runner = AgentRunner(
        registry,
        chat_executor_fallback=_fake_chat_executor,
        max_steps=4,
    )

    return SimpleNamespace(
        documents=[object()],
        chat_executor=_fake_chat_executor,
        execution_mode="test-mode",
        vector_document_count=1,
        tool_registry=registry,
        agent_runner=agent_runner,
    )


@pytest.fixture
def client():
    from backend.api import dependencies as deps_module
    from backend.api.app import app
    from backend.runtime import create_demo_runtime
    from backend.storage import history as history_module

    runtime = _build_runtime()
    deps_module.set_runtime_factory(lambda: runtime)
    history_module.set_history_store(history_module.MemoryHistoryStore())

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        deps_module.reset_runtime()
        deps_module.set_runtime_factory(create_demo_runtime)
        history_module.reset_history_store()


def test_chat_default_mode_is_rag(client):
    """默认 mode=rag：直接走 chat_executor，response 不带 decision_trace。"""
    response = client.post(
        "/chat",
        json={"question": "ping", "session_id": "s-rag"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "rag-pipeline-answer:ping"
    assert body["mode"] == "rag"
    assert body["decision_trace"] is None
    assert body["fallback"] is None


def test_chat_agent_mode_routes_through_runner(client, monkeypatch):
    """mode=agent + 知识问题：模型先调 rag_answer，再产出最终答复。"""
    planner_responses = iter(
        [
            {
                "stop_reason": "tool_use",
                "text": "",
                "tool_uses": [
                    {
                        "tool_use_id": "tu-x",
                        "name": "rag_answer",
                        "input": {"question": "policy?"},
                    }
                ],
                "raw_content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu-x",
                            "name": "rag_answer",
                            "input": {"question": "policy?"},
                        }
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "text": "Per the docs: it's allowed.",
                "tool_uses": [],
                "raw_content": [{"text": "Per the docs: it's allowed."}],
            },
        ]
    )

    monkeypatch.setattr(
        "backend.agent.planner.chat_completion_with_tools",
        lambda *args, **kwargs: next(planner_responses),
    )

    response = client.post(
        "/chat",
        json={"question": "policy?", "session_id": "s-agent", "mode": "agent"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "agent"
    assert body["answer"] == "Per the docs: it's allowed."
    assert body["fallback"] is False
    assert isinstance(body["decision_trace"], list)
    assert len(body["decision_trace"]) == 2
    assert body["decision_trace"][0]["tool_call"]["name"] == "rag_answer"
    # rag_answer surfaced its sources up to the response.
    assert body["sources"] == [
        {"rank": 1, "content": "kb-snippet", "metadata": {"source": "kb"}}
    ]


def test_chat_agent_mode_falls_back_when_planner_fails(client, monkeypatch):
    """planner 抛错时，路由仍返回 200，body.fallback=True。"""

    def boom(*args, **kwargs):
        raise RuntimeError("aws is sad")

    monkeypatch.setattr(
        "backend.agent.planner.chat_completion_with_tools",
        boom,
    )

    response = client.post(
        "/chat",
        json={
            "question": "anything",
            "session_id": "s-fallback",
            "mode": "agent",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "agent"
    assert body["fallback"] is True
    # Fallback returns the rag-pipeline answer.
    assert body["answer"] == "rag-pipeline-answer:anything"
