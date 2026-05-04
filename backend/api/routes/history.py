"""Session history routes."""

from fastapi import APIRouter

from backend.api.schemas import ClearHistoryResponse, HistoryResponse
from backend.storage.history import clear_session_history, read_session_history

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    """Read history records for the specified session."""
    messages = read_session_history(session_id)
    return HistoryResponse(session_id=session_id, messages=messages)


@router.delete("/{session_id}", response_model=ClearHistoryResponse)
def delete_history(session_id: str) -> ClearHistoryResponse:
    """Clear history records for the specified session."""
    clear_session_history(session_id)
    return ClearHistoryResponse(message="Session history cleared.", session_id=session_id)
