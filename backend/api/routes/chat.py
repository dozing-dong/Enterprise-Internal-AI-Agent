"""聊天相关路由。"""

from typing import Annotated

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends

from backend.api.dependencies import get_runtime
from backend.api.exceptions import RagException
from backend.api.schemas import ChatRequest, ChatResponse, SourceItem
from backend.runtime import DemoRuntime
from backend.storage.history import build_history_path

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    runtime: Annotated[DemoRuntime, Depends(get_runtime)],
) -> ChatResponse:
    """接收用户问题，执行 RAG 流程并返回答案与检索来源。"""
    try:
        result = await run_in_threadpool(
            runtime.chat_executor,
            request.question,
            request.session_id,
        )
    except RuntimeError as exc:
        raise RagException(str(exc), status_code=503) from exc
    except Exception as exc:
        raise RagException(f"RAG 执行时发生内部错误：{exc}", status_code=500) from exc

    history_file = build_history_path(request.session_id)

    return ChatResponse(
        answer=result["answer"],
        original_question=result["original_question"],
        retrieval_question=result["retrieval_question"],
        session_id=request.session_id,
        history_file=str(history_file),
        sources=[SourceItem(**s) for s in result["sources"]],
    )
