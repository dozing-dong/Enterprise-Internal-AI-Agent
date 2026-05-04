"""Session metadata routes.

Provides every capability the frontend sidebar needs:
- POST   /sessions              create (server-generated UUID)
- GET    /sessions              list
- PATCH  /sessions/{session_id} rename
- DELETE /sessions/{session_id} delete (cascades to history cleanup)
"""

from fastapi import APIRouter

from backend.api.exceptions import RagException
from backend.api.schemas import (
    CreateSessionRequest,
    DeleteSessionResponse,
    RenameSessionRequest,
    SessionItem,
    SessionListResponse,
)
from backend.storage.sessions import (
    DEFAULT_SESSION_TITLE,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    rename_session,
)


router = APIRouter(prefix="/sessions", tags=["Sessions"])


def _to_item(record) -> SessionItem:
    return SessionItem(
        session_id=record.session_id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post("", response_model=SessionItem, status_code=201)
def create_new_session(request: CreateSessionRequest | None = None) -> SessionItem:
    """Create a new session; returns metadata with a server-generated session_id."""
    title = (request.title if request and request.title else DEFAULT_SESSION_TITLE).strip()
    if not title:
        title = DEFAULT_SESSION_TITLE
    record = create_session(title=title)
    return _to_item(record)


@router.get("", response_model=SessionListResponse)
def list_all_sessions() -> SessionListResponse:
    """List all sessions ordered by updated_at descending."""
    records = list_sessions()
    return SessionListResponse(sessions=[_to_item(record) for record in records])


@router.patch("/{session_id}", response_model=SessionItem)
def rename_existing_session(session_id: str, request: RenameSessionRequest) -> SessionItem:
    """Rename the specified session."""
    title = request.title.strip()
    if not title:
        raise RagException("title cannot be empty.", status_code=400)
    record = rename_session(session_id, title)
    if record is None:
        raise RagException(f"Session not found: {session_id}", status_code=404)
    return _to_item(record)


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
def delete_existing_session(session_id: str) -> DeleteSessionResponse:
    """Delete session metadata and clear associated history."""
    if get_session(session_id) is None:
        raise RagException(f"Session not found: {session_id}", status_code=404)
    delete_session(session_id)
    return DeleteSessionResponse(message="Session deleted.", session_id=session_id)
