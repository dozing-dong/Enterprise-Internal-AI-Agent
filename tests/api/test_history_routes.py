"""``/chat`` 与 ``/history/{session_id}`` 的端到端回归。

通过：
- 注入 fake runtime 让 ``/chat`` 不触发真实的 RAG 链路；
- 注入 ``MemoryHistoryStore`` 让历史读写不依赖 PostgreSQL；
模拟一次完整对话：写入历史 -> 读取 -> 清空。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient


def _build_fake_runtime():
    """构造一个会主动写历史的 fake runtime。

    真实的 LangGraph executor 会在执行过程中写历史，这里复用
    ``append_session_messages`` 模拟同样的副作用，让 ``/history``
    路由能读到对应记录。
    """

    from backend.storage.history import append_session_messages

    def fake_chat(question: str, session_id: str) -> dict[str, Any]:
        append_session_messages(
            session_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"fake-answer:{question}"},
            ],
        )
        return {
            "answer": f"fake-answer:{question}",
            "original_question": question,
            "retrieval_question": question,
            "sources": [
                {"rank": 1, "content": "snippet", "metadata": {"source": "fake"}},
            ],
        }

    return SimpleNamespace(
        documents=[object()],
        chat_executor=fake_chat,
        execution_mode="fake-mode",
        vector_document_count=1,
    )


@pytest.fixture
def client():
    from backend.api import dependencies as deps_module
    from backend.api.app import app
    from backend.runtime import create_demo_runtime
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())
    deps_module.set_runtime_factory(_build_fake_runtime)

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        deps_module.reset_runtime()
        deps_module.set_runtime_factory(create_demo_runtime)
        history_module.reset_history_store()


def test_chat_uses_memory_history_locator(client):
    response = client.post(
        "/chat",
        json={"question": "ping", "session_id": "session-history-001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "session-history-001"
    assert body["history_file"].startswith("memory://chat_history/")
    assert body["history_file"].endswith("session-history-001")


def test_history_round_trip(client):
    session_id = "session-roundtrip"

    client.post(
        "/chat",
        json={"question": "Q1", "session_id": session_id},
    )
    client.post(
        "/chat",
        json={"question": "Q2", "session_id": session_id},
    )

    history_response = client.get(f"/history/{session_id}")
    assert history_response.status_code == 200
    body = history_response.json()
    assert body["session_id"] == session_id
    assert body["messages"] == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "fake-answer:Q1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "fake-answer:Q2"},
    ]


def test_history_clear(client):
    session_id = "session-to-clear"

    client.post(
        "/chat",
        json={"question": "hi", "session_id": session_id},
    )

    delete_response = client.delete(f"/history/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["session_id"] == session_id

    after = client.get(f"/history/{session_id}")
    assert after.status_code == 200
    assert after.json()["messages"] == []
