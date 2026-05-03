"""聊天相关路由。

包含两个入口：
- ``POST /chat``           兼容性入口，内部消费一次 ``stream`` 后聚合返回。
- ``POST /chat/stream``    SSE 流式接口，仅下发 token / sources / done / error。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_runtime
from backend.api.exceptions import RagException
from backend.api.schemas import ChatRequest, ChatResponse, ChatStreamRequest
from backend.orchestrator import ChatOrchestrator
from backend.runtime import DemoRuntime
from backend.storage.history import read_session_history
from backend.storage.sessions import create_session_if_missing


router = APIRouter(tags=["Chat"])


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


def _format_sse(event: str, data: dict[str, Any]) -> str:
    """按 SSE 协议拼装单条事件。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _ensure_graph(runtime: DemoRuntime, mode: str) -> None:
    if mode == "agent" and runtime.agent_graph is None:
        raise RagException("Agent graph 未初始化。", status_code=503)
    if mode == "rag" and runtime.rag_graph is None:
        raise RagException("RAG graph 未初始化。", status_code=503)
    if mode == "multi_agent" and runtime.multi_agent_graph is None:
        raise RagException(
            "Multi-Agent graph 未初始化（可能 langchain-mcp-adapters 未安装或 MCP server 启动失败）。",
            status_code=503,
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    runtime: Annotated[DemoRuntime, Depends(get_runtime)],
) -> ChatResponse:
    """非流式入口：内部消费同一条 stream 后聚合返回。

    主流业务请使用 ``POST /chat/stream``。
    """
    _ensure_graph(runtime, request.mode)
    orchestrator = ChatOrchestrator(runtime)
    return await run_in_threadpool(orchestrator.aggregate, request)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    runtime: Annotated[DemoRuntime, Depends(get_runtime)],
) -> StreamingResponse:
    """SSE 流式聊天接口。

    事件类型：``token`` / ``sources`` / ``done`` / ``error``。
    出错时统一通过 ``error`` 事件回传。
    """
    if not _SESSION_ID_RE.match(request.session_id):
        raise RagException(
            "Invalid session_id format. Expect 8-64 chars of [A-Za-z0-9_-].",
            status_code=400,
        )

    _ensure_graph(runtime, request.mode)

    session_record = create_session_if_missing(request.session_id)
    existing_history = read_session_history(request.session_id)
    is_first_turn = len(existing_history) == 0

    orchestrator = ChatOrchestrator(runtime)

    def event_stream() -> Iterator[str]:
        try:
            for event in orchestrator.stream(
                request,
                session_record_title=session_record.title,
                is_first_turn=is_first_turn,
            ):
                yield _format_sse(event.type, event.data)
        except Exception as exc:  # noqa: BLE001
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
