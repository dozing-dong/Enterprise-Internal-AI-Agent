"""会话历史存储。

设计要点：
- 通过 ``HistoryStore`` 协议抽象不同后端，避免上层耦合具体实现。
- 默认使用 PostgreSQL（与 pgvector 共用同一个数据库连接配置）。
- ``MemoryHistoryStore`` 仅用于单元测试隔离，不作为生产后端。
- 顶层函数 ``read_session_history`` / ``append_session_messages`` /
  ``clear_session_history`` / ``build_history_path`` 保持原签名不变，
  内部通过 ``_resolve_store()`` 路由到当前激活的 Store。
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from backend.config import (
    HISTORY_BACKEND,
    HISTORY_TABLE,
    PGVECTOR_CONNECTION,
)


def _normalize_message(message: dict[str, Any]) -> dict[str, str] | None:
    """统一成 {role, content}，仅接受当前项目标准消息格式。"""
    role = message.get("role")
    content = message.get("content")

    if isinstance(role, str) and isinstance(content, str):
        return {"role": role, "content": content}

    return None


def _normalize_messages(messages: list[dict]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = _normalize_message(message)
        if item is not None:
            normalized.append(item)
    return normalized


class HistoryStore(Protocol):
    """会话历史存储协议。"""

    def read(self, session_id: str) -> list[dict[str, str]]: ...

    def append(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[dict[str, str]]: ...

    def clear(self, session_id: str) -> None: ...

    def locator(self, session_id: str) -> str: ...


class MemoryHistoryStore:
    """内存实现，主要用于单元测试隔离。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def read(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[dict[str, str]]:
        normalized = _normalize_messages(messages)
        with self._lock:
            current = self._sessions.setdefault(session_id, [])
            current.extend(normalized)
            return list(current)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = []

    def locator(self, session_id: str) -> str:
        return f"memory://chat_history/{session_id}"


def _normalize_pg_connection_string(connection: str) -> str:
    """把 SQLAlchemy 风格连接串转成 psycopg 直接可用的格式。"""
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)


class PostgresHistoryStore:
    """基于 PostgreSQL 的会话历史实现。

    表结构（启动时通过 ``CREATE TABLE IF NOT EXISTS`` 建立）：
    - id          BIGSERIAL PRIMARY KEY
    - session_id  TEXT NOT NULL
    - role        TEXT NOT NULL
    - content     TEXT NOT NULL
    - created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    索引：(session_id, created_at, id)
    """

    def __init__(
        self,
        *,
        connection_string: str = PGVECTOR_CONNECTION,
        table_name: str = HISTORY_TABLE,
    ) -> None:
        self._connection_string = connection_string
        self._table_name = table_name
        self._table_initialized = False
        self._init_lock = threading.Lock()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError("缺少 psycopg 依赖，无法使用 PostgreSQL 历史存储。") from exc
        return psycopg.connect(_normalize_pg_connection_string(self._connection_string))

    def _ensure_table(self, connection) -> None:
        if self._table_initialized:
            return

        with self._init_lock:
            if self._table_initialized:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_session
                    ON {self._table_name} (session_id, created_at, id);
                    """
                )
            connection.commit()
            self._table_initialized = True

    def read(self, session_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT role, content
                    FROM {self._table_name}
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC;
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()

        return [{"role": row[0], "content": row[1]} for row in rows]

    def append(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[dict[str, str]]:
        normalized = _normalize_messages(messages)

        with self._connect() as connection:
            self._ensure_table(connection)

            if normalized:
                payload = [
                    (session_id, message["role"], message["content"])
                    for message in normalized
                ]
                with connection.cursor() as cursor:
                    cursor.executemany(
                        f"""
                        INSERT INTO {self._table_name} (session_id, role, content)
                        VALUES (%s, %s, %s);
                        """,
                        payload,
                    )
                connection.commit()

            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT role, content
                    FROM {self._table_name}
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC;
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()

        return [{"role": row[0], "content": row[1]} for row in rows]

    def clear(self, session_id: str) -> None:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._table_name} WHERE session_id = %s;",
                    (session_id,),
                )
            connection.commit()

    def locator(self, session_id: str) -> str:
        return f"postgres://{self._table_name}/{session_id}"


def _build_default_store() -> HistoryStore:
    backend = (HISTORY_BACKEND or "postgres").lower()

    if backend == "postgres":
        return PostgresHistoryStore()
    if backend == "memory":
        return MemoryHistoryStore()

    raise ValueError(
        f"未知的 HISTORY_BACKEND={HISTORY_BACKEND!r}，"
        "可选值：postgres / memory。"
    )


_current_store: HistoryStore | None = None
_store_lock = threading.Lock()


def set_history_store(store: HistoryStore) -> None:
    """主要给测试用：手动注入一个存储实现。"""
    global _current_store
    with _store_lock:
        _current_store = store


def reset_history_store() -> None:
    """清空当前注册的存储，下次调用时按配置重新创建。"""
    global _current_store
    with _store_lock:
        _current_store = None


def _resolve_store() -> HistoryStore:
    global _current_store
    if _current_store is None:
        with _store_lock:
            if _current_store is None:
                _current_store = _build_default_store()
    return _current_store


def read_session_history(session_id: str) -> list[dict]:
    """读取会话历史，向上层返回标准化后的消息列表。"""
    return _resolve_store().read(session_id)


def append_session_messages(session_id: str, messages: list[dict]) -> list[dict]:
    """追加消息并返回最新历史。"""
    return _resolve_store().append(session_id, messages)


def clear_session_history(session_id: str) -> None:
    """清空指定会话的历史。"""
    _resolve_store().clear(session_id)


def build_history_path(session_id: str) -> str:
    """返回当前后端下会话历史的定位字符串。

    - 数据库后端返回 ``postgres://<table>/<session_id>`` 形式的定位符；
    - 内存后端返回 ``memory://chat_history/<session_id>`` 形式的定位符。
    历史命名沿用旧名称是为了保持 ``ChatResponse.history_file`` 字段语义。
    """
    return _resolve_store().locator(session_id)
