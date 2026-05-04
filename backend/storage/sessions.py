"""Session metadata storage.

Design notes:
- Abstracts different backends behind the ``SessionStore`` protocol,
  using the same pattern as history.py.
- Uses PostgreSQL by default (sharing the same database connection
  configuration as pgvector / history).
- ``MemorySessionStore`` is for unit-test isolation only and is not a
  production backend.
- Top-level helpers ``create_session`` / ``list_sessions`` /
  ``rename_session`` / ``delete_session`` / ``touch_session`` /
  ``get_session`` keep a concise procedural API.
- delete cascades and clears ``rag_chat_history``.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from backend.config import (
    HISTORY_BACKEND,
    PGVECTOR_CONNECTION,
)
from backend.storage.history import (
    _normalize_pg_connection_string,
    clear_session_history,
)


SESSIONS_TABLE = "rag_chat_sessions"
DEFAULT_SESSION_TITLE = "New Chat"


@dataclass
class SessionRecord:
    """A single session metadata record."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SessionStore(Protocol):
    """Session metadata storage protocol."""

    def create(self, session_id: str, title: str = DEFAULT_SESSION_TITLE) -> SessionRecord: ...

    def create_if_missing(
        self, session_id: str, title: str = DEFAULT_SESSION_TITLE
    ) -> SessionRecord: ...

    def get(self, session_id: str) -> SessionRecord | None: ...

    def list(self) -> list[SessionRecord]: ...

    def rename(self, session_id: str, title: str) -> SessionRecord | None: ...

    def touch(self, session_id: str) -> None: ...

    def delete(self, session_id: str) -> None: ...


class MemorySessionStore:
    """In-memory implementation, primarily for unit-test isolation."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def create(self, session_id: str, title: str = DEFAULT_SESSION_TITLE) -> SessionRecord:
        with self._lock:
            now = self._now()
            record = SessionRecord(session_id=session_id, title=title, created_at=now, updated_at=now)
            self._sessions[session_id] = record
            return record

    def create_if_missing(
        self, session_id: str, title: str = DEFAULT_SESSION_TITLE
    ) -> SessionRecord:
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
            now = self._now()
            record = SessionRecord(session_id=session_id, title=title, created_at=now, updated_at=now)
            self._sessions[session_id] = record
            return record

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> list[SessionRecord]:
        with self._lock:
            return sorted(
                self._sessions.values(),
                key=lambda record: record.updated_at,
                reverse=True,
            )

    def rename(self, session_id: str, title: str) -> SessionRecord | None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            updated = SessionRecord(
                session_id=record.session_id,
                title=title,
                created_at=record.created_at,
                updated_at=self._now(),
            )
            self._sessions[session_id] = updated
            return updated

    def touch(self, session_id: str) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return
            self._sessions[session_id] = SessionRecord(
                session_id=record.session_id,
                title=record.title,
                created_at=record.created_at,
                updated_at=self._now(),
            )

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        clear_session_history(session_id)


class PostgresSessionStore:
    """PostgreSQL-backed session metadata implementation.

    Schema (created via ``CREATE TABLE IF NOT EXISTS`` at startup):
    - session_id  TEXT PRIMARY KEY
    - title       TEXT NOT NULL DEFAULT 'New Chat'
    - created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    - updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    Index: (updated_at DESC)
    """

    def __init__(
        self,
        *,
        connection_string: str = PGVECTOR_CONNECTION,
        table_name: str = SESSIONS_TABLE,
    ) -> None:
        self._connection_string = connection_string
        self._table_name = table_name
        self._table_initialized = False
        self._init_lock = threading.Lock()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError("psycopg is required to use PostgreSQL session storage.") from exc
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
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '{DEFAULT_SESSION_TITLE}',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_updated
                    ON {self._table_name} (updated_at DESC);
                    """
                )
            connection.commit()
            self._table_initialized = True

    def _row_to_record(self, row) -> SessionRecord:
        return SessionRecord(
            session_id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
        )

    def create(self, session_id: str, title: str = DEFAULT_SESSION_TITLE) -> SessionRecord:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self._table_name} (session_id, title)
                    VALUES (%s, %s)
                    RETURNING session_id, title, created_at, updated_at;
                    """,
                    (session_id, title),
                )
                row = cursor.fetchone()
            connection.commit()
        return self._row_to_record(row)

    def create_if_missing(
        self, session_id: str, title: str = DEFAULT_SESSION_TITLE
    ) -> SessionRecord:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self._table_name} (session_id, title)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                    RETURNING session_id, title, created_at, updated_at;
                    """,
                    (session_id, title),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        f"""
                        SELECT session_id, title, created_at, updated_at
                        FROM {self._table_name}
                        WHERE session_id = %s;
                        """,
                        (session_id,),
                    )
                    row = cursor.fetchone()
            connection.commit()
        return self._row_to_record(row)

    def get(self, session_id: str) -> SessionRecord | None:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT session_id, title, created_at, updated_at
                    FROM {self._table_name}
                    WHERE session_id = %s;
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def list(self) -> list[SessionRecord]:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT session_id, title, created_at, updated_at
                    FROM {self._table_name}
                    ORDER BY updated_at DESC, created_at DESC;
                    """
                )
                rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def rename(self, session_id: str, title: str) -> SessionRecord | None:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET title = %s, updated_at = now()
                    WHERE session_id = %s
                    RETURNING session_id, title, created_at, updated_at;
                    """,
                    (title, session_id),
                )
                row = cursor.fetchone()
            connection.commit()
        return self._row_to_record(row) if row else None

    def touch(self, session_id: str) -> None:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET updated_at = now()
                    WHERE session_id = %s;
                    """,
                    (session_id,),
                )
            connection.commit()

    def delete(self, session_id: str) -> None:
        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._table_name} WHERE session_id = %s;",
                    (session_id,),
                )
            connection.commit()
        clear_session_history(session_id)


def _build_default_store() -> SessionStore:
    backend = (HISTORY_BACKEND or "postgres").lower()

    if backend == "postgres":
        return PostgresSessionStore()
    if backend == "memory":
        return MemorySessionStore()

    raise ValueError(
        f"Unknown HISTORY_BACKEND={HISTORY_BACKEND!r}; "
        "valid values: postgres / memory."
    )


_current_store: SessionStore | None = None
_store_lock = threading.Lock()


def set_session_store(store: SessionStore) -> None:
    """Mainly for tests: manually inject a storage implementation."""
    global _current_store
    with _store_lock:
        _current_store = store


def reset_session_store() -> None:
    """Drop the currently registered store; the next call rebuilds from configuration."""
    global _current_store
    with _store_lock:
        _current_store = None


def _resolve_store() -> SessionStore:
    global _current_store
    if _current_store is None:
        with _store_lock:
            if _current_store is None:
                _current_store = _build_default_store()
    return _current_store


def generate_session_id() -> str:
    """Generate a normalized session_id (UUID4 hex, 32 characters)."""
    return uuid.uuid4().hex


def create_session(title: str = DEFAULT_SESSION_TITLE) -> SessionRecord:
    """Create a new session with an auto-generated session_id."""
    return _resolve_store().create(generate_session_id(), title)


def create_session_if_missing(
    session_id: str, title: str = DEFAULT_SESSION_TITLE
) -> SessionRecord:
    """Ensure session_id exists; used as a safety net before streaming chat."""
    return _resolve_store().create_if_missing(session_id, title)


def get_session(session_id: str) -> SessionRecord | None:
    """Read a single session metadata record."""
    return _resolve_store().get(session_id)


def list_sessions() -> list[SessionRecord]:
    """List all sessions in descending order of updated_at."""
    return _resolve_store().list()


def rename_session(session_id: str, title: str) -> SessionRecord | None:
    """Rename a session."""
    return _resolve_store().rename(session_id, title)


def touch_session(session_id: str) -> None:
    """Update updated_at so the session bubbles to the top of the list."""
    _resolve_store().touch(session_id)


def delete_session(session_id: str) -> None:
    """Delete session metadata and cascade-clean its history."""
    _resolve_store().delete(session_id)
