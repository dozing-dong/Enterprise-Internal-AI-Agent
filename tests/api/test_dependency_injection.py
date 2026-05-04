"""DI regression tests: inject a fake runtime via set_runtime_factory + dependency_overrides.

Goals:
- The /, /health, /chat endpoints never trigger real pgvector / Bedrock calls.
- Verify that the FastAPI composition root (init_runtime + get_runtime) is replaceable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


class _FakeRagGraph:
    """Synchronous rag_graph stand-in: ``.invoke`` + ``.stream``."""

    def invoke(self, payload):
        question = payload["question"]
        return {
            "answer": f"fake-answer:{question}:{payload.get('session_id', '')}",
            "sources": [
                {"rank": 1, "content": "snippet", "metadata": {"source": "fake"}},
            ],
            "retrieval_question": question,
            "original_question": question,
        }

    def stream(self, payload, *, stream_mode):
        result = self.invoke(payload)
        from langchain_core.messages import AIMessageChunk

        # Inject sources and retrieval_question in a single updates chunk.
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
        # Simulate the token chunk emitted by the final generation node.
        chunk = AIMessageChunk(content=result["answer"])
        meta = {"langgraph_node": "generate_answer"}
        yield ("messages", (chunk, meta))


def _build_fake_runtime() -> SimpleNamespace:
    """Build a minimal DemoRuntime stand-in with no external dependencies."""
    rag_graph = _FakeRagGraph()
    return SimpleNamespace(
        documents=[object()],
        execution_mode="fake-mode",
        vector_document_count=42,
        rag_graph=rag_graph,
        agent_graph=None,
    )


@pytest.fixture
def client():
    from backend.api import dependencies as deps_module
    from backend.api.app import app
    from backend.runtime import create_demo_runtime
    from backend.storage import history as history_module

    fake_runtime = _build_fake_runtime()
    deps_module.set_runtime_factory(lambda: fake_runtime)
    history_module.set_history_store(history_module.MemoryHistoryStore())

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        deps_module.reset_runtime()
        deps_module.set_runtime_factory(create_demo_runtime)
        history_module.reset_history_store()


def test_root_returns_injected_execution_mode(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["execution_mode"] == "fake-mode"


def test_health_returns_injected_counts(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["execution_mode"] == "fake-mode"
    assert body["vector_document_count"] == 42
    assert body["raw_document_count"] == 1


def test_chat_uses_injected_executor(client):
    response = client.post(
        "/chat",
        json={"question": "ping", "session_id": "s1-test1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake-answer:ping:s1-test1"
    assert body["session_id"] == "s1-test1"
    assert body["sources"][0]["metadata"]["source"] == "fake"
