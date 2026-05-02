"""聊天相关路由。

包含两个入口：
- ``POST /chat``           兼容旧版本契约，非流式，一次性返回完整答案。
- ``POST /chat/stream``    SSE 流式接口，给前端渐进式展示进度 + token + 来源。
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
from backend.api.schemas import ChatRequest, ChatResponse, ChatStreamRequest, SourceItem
from backend.config import RETRIEVER_CANDIDATE_K, RERANK_TOP_K, RETRIEVER_TOP_K
from backend.data.processing import convert_docs_to_sources, format_docs
from backend.rag.models import chat_completion_stream
from backend.rag.retrievers import fuse_retrieval_results
from backend.rag.rewrite import rewrite_question_for_retrieval
from backend.rag.title import generate_session_title
from backend.runtime import DemoRuntime
from backend.storage.history import (
    append_session_messages,
    build_history_path,
    read_session_history,
)
from backend.storage.sessions import (
    create_session_if_missing,
    rename_session,
    touch_session,
)


router = APIRouter(tags=["Chat"])


# 与前端 createSession 返回的 UUID4 hex（32 位）保持兼容，
# 同时允许 8-64 位字母数字下划线连字符，方便测试时用语义化 ID。
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")

# 与 backend/rag/chain.py 中 generate_answer 节点保持一致，
# 这样流式与非流式路径产出的回答风格统一。
_GENERATION_SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on retrieval results. "
    "Prefer to rely on the provided knowledge base snippets and the conversation history. "
    "If the reference content is not sufficient to support a conclusion, "
    "clearly say that you do not know and do not fabricate an answer."
)


def _format_sse(event: str, data: dict[str, Any]) -> str:
    """按 SSE 协议拼装单条事件。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_user_message(question: str, context: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Question: {question}\n\n"
            f"Reference knowledge:\n{context}\n\n"
            "Please answer the question based on the reference knowledge above."
        ),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    runtime: Annotated[DemoRuntime, Depends(get_runtime)],
) -> ChatResponse:
    """接收用户问题，按 mode 选择执行路径并返回答案与检索来源。

    - ``mode=rag``：保持原行为，调用编译后的 langgraph 流水线。
    - ``mode=agent``：进入 AgentRunner，由模型自决调用工具；
      任何异常会被 runner 内部兜底到 RAG 流水线。
    """
    if request.mode == "agent":
        return await _run_chat_agent(request, runtime)

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
        mode="rag",
    )


async def _run_chat_agent(
    request: ChatRequest,
    runtime: DemoRuntime,
) -> ChatResponse:
    """Agent 模式专用分支。

    Runner 内部已经做了 fallback，只有在 fallback 也失败时才会抛出，
    这里把异常映射为 RagException，与 rag 模式保持一致的错误语义。
    """
    if runtime.agent_runner is None:
        # 一般不会发生（runtime 启动时构建过），保险起见做显式提示。
        raise RagException("Agent runner 未初始化。", status_code=503)

    try:
        run_result = await run_in_threadpool(
            runtime.agent_runner.run,
            request.question,
            request.session_id,
        )
    except RuntimeError as exc:
        raise RagException(str(exc), status_code=503) from exc
    except Exception as exc:
        raise RagException(
            f"Agent 执行时发生内部错误：{exc}", status_code=500
        ) from exc

    history_file = build_history_path(request.session_id)

    return ChatResponse(
        answer=run_result.answer,
        original_question=request.question,
        retrieval_question=run_result.retrieval_question or request.question,
        session_id=request.session_id,
        history_file=str(history_file),
        sources=[SourceItem(**s) for s in run_result.sources],
        decision_trace=run_result.decision_trace,
        fallback=run_result.fallback,
        mode="agent",
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    runtime: Annotated[DemoRuntime, Depends(get_runtime)],
) -> StreamingResponse:
    """SSE 流式聊天接口。

    根据 ``request.mode`` 分发：
    - ``rag``：保留原事件序列
      ``progress(rewriting) -> progress(retrieving) -> sources
        -> progress(generating) -> token * N -> done``。
    - ``agent``：基于 AgentRunner 的多步循环，事件序列形如
      ``progress(deciding) -> [tool_call -> tool_result -> sources?]*
        -> token * N -> done``。

    出错时统一通过 ``error`` 事件回传。
    """
    if not _SESSION_ID_RE.match(request.session_id):
        raise RagException(
            "Invalid session_id format. Expect 8-64 chars of [A-Za-z0-9_-].",
            status_code=400,
        )

    session_record = create_session_if_missing(request.session_id)
    existing_history = read_session_history(request.session_id)
    is_first_turn = len(existing_history) == 0

    if request.mode == "agent":
        return _build_agent_stream_response(
            request,
            runtime,
            session_record_title=session_record.title,
            is_first_turn=is_first_turn,
        )

    rewrite_chain = runtime.rewrite_chain
    vector_retriever = runtime.vector_retriever
    keyword_retriever = runtime.keyword_retriever
    reranker = runtime.reranker

    def event_stream() -> Iterator[str]:
        try:
            yield _format_sse(
                "progress",
                {"stage": "rewriting", "message": "Refining your query..."},
            )
            retrieval_question = rewrite_question_for_retrieval(
                request.question,
                rewrite_chain,
            )

            yield _format_sse(
                "progress",
                {"stage": "retrieving", "message": "Searching the knowledge base..."},
            )
            vector_docs = vector_retriever.invoke(retrieval_question)
            keyword_docs = keyword_retriever.invoke(retrieval_question)
            candidate_top_k = (
                max(RETRIEVER_CANDIDATE_K, RERANK_TOP_K)
                if reranker is not None
                else RETRIEVER_TOP_K
            )
            fused_docs = fuse_retrieval_results(
                vector_docs,
                keyword_docs,
                top_k=candidate_top_k,
            )

            final_docs = fused_docs
            if reranker is not None and fused_docs:
                yield _format_sse(
                    "progress",
                    {"stage": "reranking", "message": "Reranking retrieved snippets..."},
                )
                final_docs = reranker.invoke(
                    retrieval_question,
                    fused_docs,
                    RERANK_TOP_K,
                )

            sources = convert_docs_to_sources(final_docs)
            context = format_docs(final_docs)

            yield _format_sse(
                "sources",
                {
                    "sources": sources,
                    "retrieval_question": retrieval_question,
                    "original_question": request.question,
                },
            )

            yield _format_sse(
                "progress",
                {"stage": "generating", "message": "Generating answer..."},
            )

            messages = list(existing_history)
            messages.append(_build_user_message(request.question, context))

            answer_parts: list[str] = []
            for token in chat_completion_stream(
                messages,
                system_prompt=_GENERATION_SYSTEM_PROMPT,
                temperature=0.0,
            ):
                answer_parts.append(token)
                yield _format_sse("token", {"text": token})

            full_answer = "".join(answer_parts).strip() or "No answer generated."

            append_session_messages(
                request.session_id,
                [
                    {"role": "user", "content": request.question},
                    {"role": "assistant", "content": full_answer},
                ],
            )
            touch_session(request.session_id)

            title = session_record.title
            if is_first_turn:
                yield _format_sse(
                    "progress",
                    {"stage": "titling", "message": "Naming this chat..."},
                )
                try:
                    new_title = generate_session_title(request.question, full_answer)
                    if new_title:
                        renamed = rename_session(request.session_id, new_title)
                        if renamed is not None:
                            title = renamed.title
                except Exception:
                    # 标题生成失败不应影响主流程，保留默认 'New Chat'。
                    pass

            yield _format_sse(
                "done",
                {
                    "session_id": request.session_id,
                    "title": title,
                    "full_answer": full_answer,
                    "original_question": request.question,
                    "retrieval_question": retrieval_question,
                    "mode": "rag",
                },
            )
        except Exception as exc:  # noqa: BLE001 - 任何异常都通过 SSE error 事件回传
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # 关闭 nginx / 代理层的缓冲，确保 token 实时下推。
            "X-Accel-Buffering": "no",
        },
    )


def _build_agent_stream_response(
    request: ChatStreamRequest,
    runtime: DemoRuntime,
    *,
    session_record_title: str,
    is_first_turn: bool,
) -> StreamingResponse:
    """组装 agent 模式下的 SSE 响应。

    与 rag 模式不同点：
    - 不在路由里组装 retriever / reranker，AgentRunner 内部通过 rag_answer 工具完成。
    - 历史落库由 AgentRunner 自己处理，路由层只需补 ``touch_session`` 与 ``done``。
    """
    if runtime.agent_runner is None:
        raise RagException("Agent runner 未初始化。", status_code=503)

    agent_runner = runtime.agent_runner

    def event_stream() -> Iterator[str]:
        try:
            done_payload: dict[str, Any] | None = None

            for event in agent_runner.run_stream(
                request.question,
                request.session_id,
            ):
                if event.type == "done":
                    # 暂存 done，让我们能在末尾合并 title / mode 后再下发。
                    done_payload = dict(event.data)
                    continue

                yield _format_sse(event.type, event.data)

            touch_session(request.session_id)

            full_answer = (done_payload or {}).get("full_answer", "")
            title = session_record_title
            if is_first_turn and full_answer:
                yield _format_sse(
                    "progress",
                    {"stage": "titling", "message": "Naming this chat..."},
                )
                try:
                    new_title = generate_session_title(
                        request.question, full_answer
                    )
                    if new_title:
                        renamed = rename_session(
                            request.session_id, new_title
                        )
                        if renamed is not None:
                            title = renamed.title
                except Exception:
                    # 标题生成不阻塞主流程。
                    pass

            final_done: dict[str, Any] = {
                "session_id": request.session_id,
                "title": title,
                "full_answer": full_answer,
                "original_question": request.question,
                "retrieval_question": (done_payload or {}).get(
                    "retrieval_question", request.question
                ),
                "decision_trace": (done_payload or {}).get(
                    "decision_trace", []
                ),
                "fallback": (done_payload or {}).get("fallback", False),
                "mode": "agent",
            }
            yield _format_sse("done", final_done)
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
