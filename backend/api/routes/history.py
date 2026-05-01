"""会话历史相关路由。"""

from fastapi import APIRouter

from backend.api.schemas import ClearHistoryResponse, HistoryResponse
from backend.storage.history import clear_session_history, read_session_history

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    """读取指定会话的历史记录。"""
    messages = read_session_history(session_id)
    return HistoryResponse(session_id=session_id, messages=messages)


@router.delete("/{session_id}", response_model=ClearHistoryResponse)
def delete_history(session_id: str) -> ClearHistoryResponse:
    """清空指定会话的历史记录。"""
    clear_session_history(session_id)
    return ClearHistoryResponse(message="会话历史已清空。", session_id=session_id)
