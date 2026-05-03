"""``/chat`` 与 ``/history/{session_id}`` 的端到端回归。

通过：
- 注入 fake runtime 让 ``/chat`` 不触发真实的 RAG 链路；
- 注入 ``MemoryHistoryStore`` 让历史读写不依赖 PostgreSQL；
模拟一次完整对话：写入历史 -> 读取 -> 清空。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


class _FakeRagGraph:
    """fake rag_graph：``.invoke`` 真正写历史；``.stream`` 复用之，并产出 token。"""

    def invoke(self, payload):
        from backend.storage.history import append_session_messages

        question = payload["question"]
        session_id = payload["session_id"]
        answer = f"fake-answer:{question}"
        append_session_messages(
            session_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )
        return {
            "answer": answer,
            "sources": [
                {"rank": 1, "content": "snippet", "metadata": {"source": "fake"}},
            ],
            "retrieval_question": question,
            "original_question": question,
        }

    def stream(self, payload, *, stream_mode):
        result = self.invoke(payload)
        from langchain_core.messages import AIMessageChunk

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
        chunk = AIMessageChunk(content=result["answer"])
        meta = {"langgraph_node": "generate_answer"}
        yield ("messages", (chunk, meta))


def _build_fake_runtime():
    rag_graph = _FakeRagGraph()
    return SimpleNamespace(
        documents=[object()],
        execution_mode="fake-mode",
        vector_document_count=1,
        rag_graph=rag_graph,
        agent_graph=None,
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
