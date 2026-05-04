"""Session history storage.

Design notes:
- Abstracts different backends behind the ``HistoryStore`` protocol so
  the upper layers do not couple to a concrete implementation.
- Uses PostgreSQL by default (sharing the same database connection
  configuration as pgvector).
- ``MemoryHistoryStore`` is for unit-test isolation only and is not a
  production backend.
- Top-level helpers ``read_session_history`` /
  ``append_session_messages`` / ``clear_session_history`` /
  ``build_history_path`` keep their original signatures and route
  internally through ``_resolve_store()`` to the active store.
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
    """Normalize to {role, content}; only the project-standard message format is accepted."""
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
    """Session history storage protocol."""

    def read(self, session_id: str) -> list[dict[str, str]]: ...

    def append(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[dict[str, str]]: ...

    def clear(self, session_id: str) -> None: ...

    def locator(self, session_id: str) -> str: ...


class MemoryHistoryStore:
    """In-memory implementation, primarily for unit-test isolation."""

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
    """Convert SQLAlchemy-style connection strings to a directly psycopg-compatible format."""
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)


class PostgresHistoryStore:
    """PostgreSQL-backed session history implementation.

    Schema (created via ``CREATE TABLE IF NOT EXISTS`` at startup):
    - id          BIGSERIAL PRIMARY KEY
    - session_id  TEXT NOT NULL
    - role        TEXT NOT NULL
    - content     TEXT NOT NULL
    - created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    Index: (session_id, created_at, id)
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
            raise ImportError("psycopg is required to use PostgreSQL history storage.") from exc
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
        f"Unknown HISTORY_BACKEND={HISTORY_BACKEND!r}; "
        "valid values: postgres / memory."
    )


_current_store: HistoryStore | None = None
_store_lock = threading.Lock()


def set_history_store(store: HistoryStore) -> None:
    """Mainly for tests: manually inject a storage implementation."""
    global _current_store
    with _store_lock:
        _current_store = store


def reset_history_store() -> None:
    """Drop the currently registered store; the next call rebuilds from configuration."""
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
    """Read session history; returns the normalized message list to upper layers."""
    return _resolve_store().read(session_id)


def append_session_messages(session_id: str, messages: list[dict]) -> list[dict]:
    """Append messages and return the latest history."""
    return _resolve_store().append(session_id, messages)


def clear_session_history(session_id: str) -> None:
    """Clear the history for the given session."""
    _resolve_store().clear(session_id)


def build_history_path(session_id: str) -> str:
    """Return a locator string for the session history under the current backend.

    - Database backends return locators of the form ``postgres://<table>/<session_id>``.
    - The memory backend returns locators of the form ``memory://chat_history/<session_id>``.
    The historical name is preserved to keep the semantics of
    ``ChatResponse.history_file`` stable.
    """
    return _resolve_store().locator(session_id)
