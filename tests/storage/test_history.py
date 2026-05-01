"""会话历史存储的单元测试。

只针对 ``MemoryHistoryStore`` 这条无外部依赖路径，覆盖：
读空 / 追加顺序 / 清空 / locator / 默认后端选择。
"""

from __future__ import annotations

import pytest

from backend.storage.history import (
    MemoryHistoryStore,
    _build_default_store,
    _normalize_messages,
    append_session_messages,
    build_history_path,
    clear_session_history,
    read_session_history,
    reset_history_store,
    set_history_store,
)


@pytest.fixture(autouse=True)
def memory_backend():
    """每个用例都使用全新的 MemoryHistoryStore，避免跨用例污染。"""
    set_history_store(MemoryHistoryStore())
    try:
        yield
    finally:
        reset_history_store()


def test_read_empty_session_returns_empty_list():
    assert read_session_history("not-exist") == []


def test_append_preserves_order_and_returns_full_history():
    first_batch = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    second_batch = [
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]

    after_first = append_session_messages("s-order", first_batch)
    after_second = append_session_messages("s-order", second_batch)

    assert after_first == first_batch
    assert after_second == first_batch + second_batch
    assert read_session_history("s-order") == first_batch + second_batch


def test_append_filters_invalid_messages():
    messages = [
        {"role": "user", "content": "ok"},
        {"role": "user"},
        "not-a-dict",
        {"role": 1, "content": "bad-role"},
        {"role": "assistant", "content": "fine"},
    ]
    history = append_session_messages("s-filter", messages)

    assert history == [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "fine"},
    ]


def test_clear_resets_history():
    append_session_messages(
        "s-clear",
        [{"role": "user", "content": "hello"}],
    )
    assert read_session_history("s-clear") != []

    clear_session_history("s-clear")
    assert read_session_history("s-clear") == []


def test_locator_uses_active_backend():
    locator = build_history_path("s-locator")
    assert locator.startswith("memory://chat_history/")
    assert locator.endswith("s-locator")


def test_normalize_messages_rejects_non_project_payload():
    payload = [
        {"type": "human", "data": {"content": "你好"}},
        {"type": "ai", "data": {"content": "你好，请问你是？"}},
        {"type": "system", "data": {"content": "ignore"}},
    ]
    normalized = _normalize_messages(payload)
    assert normalized == []


def test_build_default_store_supports_known_backends(monkeypatch):
    import backend.storage.history as history_module

    monkeypatch.setattr(history_module, "HISTORY_BACKEND", "memory")
    assert isinstance(_build_default_store(), MemoryHistoryStore)


def test_build_default_store_rejects_unknown_backend(monkeypatch):
    import backend.storage.history as history_module

    monkeypatch.setattr(history_module, "HISTORY_BACKEND", "redis")
    with pytest.raises(ValueError):
        _build_default_store()
