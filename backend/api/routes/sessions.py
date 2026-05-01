"""会话元数据相关路由。

提供前端侧边栏需要的全部能力：
- POST   /sessions              新建（自动生成 UUID）
- GET    /sessions              列表
- PATCH  /sessions/{session_id} 重命名
- DELETE /sessions/{session_id} 删除（级联清理历史）
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
    """创建一条新会话；返回带服务端生成 session_id 的元数据。"""
    title = (request.title if request and request.title else DEFAULT_SESSION_TITLE).strip()
    if not title:
        title = DEFAULT_SESSION_TITLE
    record = create_session(title=title)
    return _to_item(record)


@router.get("", response_model=SessionListResponse)
def list_all_sessions() -> SessionListResponse:
    """按 updated_at 倒序列出全部会话。"""
    records = list_sessions()
    return SessionListResponse(sessions=[_to_item(record) for record in records])


@router.patch("/{session_id}", response_model=SessionItem)
def rename_existing_session(session_id: str, request: RenameSessionRequest) -> SessionItem:
    """重命名指定会话。"""
    title = request.title.strip()
    if not title:
        raise RagException("title 不能为空。", status_code=400)
    record = rename_session(session_id, title)
    if record is None:
        raise RagException(f"会话不存在：{session_id}", status_code=404)
    return _to_item(record)


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
def delete_existing_session(session_id: str) -> DeleteSessionResponse:
    """删除会话元数据并清空对应历史。"""
    if get_session(session_id) is None:
        raise RagException(f"会话不存在：{session_id}", status_code=404)
    delete_session(session_id)
    return DeleteSessionResponse(message="Session deleted.", session_id=session_id)
