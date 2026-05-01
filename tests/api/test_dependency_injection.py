"""DI 回归测试：通过 set_runtime_factory + dependency_overrides 注入 fake runtime。

目标：
- /、/health、/chat 三个端点全程不触发真实 pgvector / Bedrock 调用。
- 验证 FastAPI 组合根（init_runtime + get_runtime）行为可被替换。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient


def _build_fake_runtime() -> SimpleNamespace:
    """构造一个无外部依赖的最小 DemoRuntime 替身。"""

    def fake_chat(question: str, session_id: str) -> dict[str, Any]:
        return {
            "answer": f"fake-answer:{question}:{session_id}",
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
        vector_document_count=42,
    )


@pytest.fixture
def client():
    from backend.api import dependencies as deps_module
    from backend.api.app import app
    from backend.runtime import create_demo_runtime

    fake_runtime = _build_fake_runtime()
    deps_module.set_runtime_factory(lambda: fake_runtime)

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        deps_module.reset_runtime()
        deps_module.set_runtime_factory(create_demo_runtime)


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
        json={"question": "ping", "session_id": "s1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake-answer:ping:s1"
    assert body["session_id"] == "s1"
    assert body["sources"][0]["metadata"]["source"] == "fake"
