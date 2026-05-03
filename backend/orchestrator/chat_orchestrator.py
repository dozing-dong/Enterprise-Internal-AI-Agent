"""Controller 下层编排：单一 stream() + aggregate()，全 LangGraph + 全流式。

设计要点：
- ``stream(...)``：按 mode 选择 RAG / Agent 编译图，统一驱动
  ``graph.stream(stream_mode=["messages","updates"])``：
  - ``messages`` 流：只转发"用户可见"模型节点的文本 chunk（按 tag 过滤），
    避免 Agent 调 RAG 工具时 RAG 内部的 ``generate_answer`` 也被双发。
  - ``updates`` 流：用 ``TraceCollector`` 累积统一 trace；如果某节点写入
    ``sources`` / ``retrieval_question``，立即下发一次 ``sources`` 事件。
- 末尾负责 ``touch_session`` / 首回合标题生成 / 下发 ``done``。
- ``aggregate(...)``：``POST /chat`` 用，内部直接消费 ``stream``，把 token 拼成
  完整 ``answer``，返回 ``ChatResponse``。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple

from backend.agent.graph import AGENT_MAIN_TAG, build_initial_messages
from backend.api.exceptions import RagException
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
    SourceItem,
)
from backend.orchestrator.trace import TraceCollector
from backend.rag.title import generate_session_title
from backend.runtime import DemoRuntime
from backend.storage.history import build_history_path, read_session_history
from backend.storage.sessions import (
    create_session_if_missing,
    rename_session,
    touch_session,
)


# RAG 节点中 "用户可见" 的最终生成节点名。orchestrator 只把该节点的
# token 转给前端，避免 Agent 调 RAG 工具时把 RAG 内部生成也下发。
_RAG_FINAL_NODE_NAME = "generate_answer"


class OrchestratorStreamEvent(NamedTuple):
    type: str
    data: dict[str, Any]


class ChatOrchestrator:
    """统一 RAG / Agent 两种模式的流式入口。"""

    def __init__(self, runtime: DemoRuntime) -> None:
        self._runtime = runtime

    # ---- stream ----------------------------------------------------------

    def stream(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
        """按 mode 选择图并驱动流式输出。"""
        if request.mode == "agent":
            yield from self._stream_agent(
                request,
                session_record_title=session_record_title,
                is_first_turn=is_first_turn,
            )
        else:
            yield from self._stream_rag(
                request,
                session_record_title=session_record_title,
                is_first_turn=is_first_turn,
            )

    # ---- aggregate (POST /chat) -----------------------------------------

    def aggregate(self, request: ChatRequest) -> ChatResponse:
        """非流式入口：消费一次 stream，把结果聚合成 ChatResponse。"""
        stream_request = ChatStreamRequest(
            question=request.question,
            session_id=request.session_id,
            mode=request.mode,
        )
        # session 元数据 / 首回合判断由 stream 入口同样需要，统一在这里准备。
        session_record = create_session_if_missing(request.session_id)
        existing_history = read_session_history(request.session_id)
        is_first_turn = len(existing_history) == 0

        token_parts: list[str] = []
        sources: list[dict] = []
        retrieval_question = request.question
        trace: list[dict] = []
        title = session_record.title
        error: str | None = None

        for event in self.stream(
            stream_request,
            session_record_title=session_record.title,
            is_first_turn=is_first_turn,
        ):
            if event.type == "token":
                token_parts.append(event.data.get("text", ""))
            elif event.type == "sources":
                sources = event.data.get("sources", []) or []
                retrieval_question = event.data.get(
                    "retrieval_question", retrieval_question
                )
            elif event.type == "done":
                trace = event.data.get("trace", []) or []
                title = event.data.get("title", title)
                full_answer = event.data.get("full_answer", "")
                if full_answer and not token_parts:
                    token_parts = [full_answer]
            elif event.type == "error":
                error = event.data.get("detail", "")
                break

        if error:
            raise RagException(
                f"{request.mode.upper()} 执行时发生内部错误：{error}",
                status_code=500,
            )

        full_answer = "".join(token_parts).strip() or "No answer generated."
        history_file = build_history_path(request.session_id)

        return ChatResponse(
            answer=full_answer,
            original_question=request.question,
            retrieval_question=retrieval_question,
            session_id=request.session_id,
            history_file=str(history_file),
            sources=[SourceItem(**s) for s in sources],
            trace=trace,
            mode=request.mode,
        )

    # ---- RAG mode --------------------------------------------------------

    def _stream_rag(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
        graph = self._runtime.rag_graph
        if graph is None:
            yield OrchestratorStreamEvent(
                "error", {"detail": "RAG graph 未初始化。"}
            )
            return

        collector = TraceCollector()
        token_parts: list[str] = []
        sources: list[dict] = []
        retrieval_question: str = request.question

        try:
            stream_iter = graph.stream(
                {
                    "question": request.question,
                    "session_id": request.session_id,
                },
                stream_mode=["messages", "updates"],
            )
            for stream_mode, payload in stream_iter:
                if stream_mode == "messages":
                    text = _extract_user_visible_text(
                        payload,
                        allowed_node=_RAG_FINAL_NODE_NAME,
                    )
                    if text:
                        token_parts.append(text)
                        yield OrchestratorStreamEvent("token", {"text": text})
                elif stream_mode == "updates":
                    new_sources, new_rq = _consume_update(
                        payload, collector
                    )
                    if new_sources is not None:
                        sources = new_sources
                    if new_rq is not None:
                        retrieval_question = new_rq
                    if new_sources is not None:
                        yield OrchestratorStreamEvent(
                            "sources",
                            {
                                "sources": sources,
                                "retrieval_question": retrieval_question,
                                "original_question": request.question,
                            },
                        )
        except Exception as exc:  # noqa: BLE001
            yield OrchestratorStreamEvent("error", {"detail": str(exc)})
            return

        full_answer = "".join(token_parts).strip() or "No answer generated."
        touch_session(request.session_id)

        title = self._maybe_retitle(
            request,
            session_record_title=session_record_title,
            is_first_turn=is_first_turn,
            full_answer=full_answer,
        )

        yield OrchestratorStreamEvent(
            "done",
            {
                "session_id": request.session_id,
                "title": title,
                "full_answer": full_answer,
                "original_question": request.question,
                "retrieval_question": retrieval_question,
                "trace": collector.to_list(),
                "mode": "rag",
            },
        )

    # ---- Agent mode ------------------------------------------------------

    def _stream_agent(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
        graph = self._runtime.agent_graph
        if graph is None:
            yield OrchestratorStreamEvent(
                "error", {"detail": "Agent graph 未初始化。"}
            )
            return

        collector = TraceCollector()
        token_parts: list[str] = []
        sources: list[dict] = []
        retrieval_question: str = request.question
        last_sources_signature: int | None = None

        existing_history = read_session_history(request.session_id)
        initial_messages = build_initial_messages(
            existing_history, request.question
        )

        try:
            stream_iter = graph.stream(
                {
                    "messages": initial_messages,
                    "session_id": request.session_id,
                    "sources": [],
                    "retrieval_question": None,
                    "original_question": request.question,
                },
                stream_mode=["messages", "updates"],
            )
            for stream_mode, payload in stream_iter:
                if stream_mode == "messages":
                    text = _extract_user_visible_text(
                        payload,
                        allowed_tag=AGENT_MAIN_TAG,
                    )
                    if text:
                        token_parts.append(text)
                        yield OrchestratorStreamEvent("token", {"text": text})
                elif stream_mode == "updates":
                    new_sources, new_rq = _consume_agent_update(
                        payload, collector
                    )
                    if new_sources is not None:
                        sources = new_sources
                    if new_rq is not None:
                        retrieval_question = new_rq
                    if new_sources is not None and id(sources) != last_sources_signature:
                        last_sources_signature = id(sources)
                        yield OrchestratorStreamEvent(
                            "sources",
                            {
                                "sources": sources,
                                "retrieval_question": retrieval_question,
                                "original_question": request.question,
                            },
                        )
        except Exception as exc:  # noqa: BLE001
            yield OrchestratorStreamEvent("error", {"detail": str(exc)})
            return

        full_answer = "".join(token_parts).strip()

        if full_answer:
            from backend.storage.history import append_session_messages

            append_session_messages(
                request.session_id,
                [
                    {"role": "user", "content": request.question},
                    {"role": "assistant", "content": full_answer},
                ],
            )

        if not full_answer:
            full_answer = "No answer generated."

        touch_session(request.session_id)

        title = self._maybe_retitle(
            request,
            session_record_title=session_record_title,
            is_first_turn=is_first_turn,
            full_answer=full_answer,
        )

        yield OrchestratorStreamEvent(
            "done",
            {
                "session_id": request.session_id,
                "title": title,
                "full_answer": full_answer,
                "original_question": request.question,
                "retrieval_question": retrieval_question,
                "trace": collector.to_list(),
                "mode": "agent",
            },
        )

    # ---- helpers ---------------------------------------------------------

    def _maybe_retitle(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
        full_answer: str,
    ) -> str:
        if not is_first_turn or not full_answer:
            return session_record_title
        try:
            new_title = generate_session_title(request.question, full_answer)
        except Exception:  # noqa: BLE001
            return session_record_title
        if not new_title:
            return session_record_title
        renamed = rename_session(request.session_id, new_title)
        if renamed is None:
            return session_record_title
        return renamed.title


# ---------------------------------------------------------------------------
# Stream payload helpers
# ---------------------------------------------------------------------------


def _extract_user_visible_text(
    payload: Any,
    *,
    allowed_node: str | None = None,
    allowed_tag: str | None = None,
) -> str:
    """从 LangGraph ``messages`` 模式的 chunk 中抽出可见文本。

    ``payload`` 为 ``(message_chunk, metadata)`` 元组。按 ``allowed_node`` 或
    ``allowed_tag`` 过滤来源，避免子图（例如 RAG-as-tool）的内部生成被
    转发给前端。
    """
    if not isinstance(payload, tuple) or len(payload) != 2:
        return ""
    message_chunk, metadata = payload

    if allowed_node is not None:
        node = (metadata or {}).get("langgraph_node")
        if node != allowed_node:
            return ""
    if allowed_tag is not None:
        tags = (metadata or {}).get("tags") or []
        if allowed_tag not in tags:
            return ""

    return _flatten_message_text(getattr(message_chunk, "content", ""))


def _flatten_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # ChatBedrockConverse 流式 chunk 形如 {"type":"text","text":"..."}
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _consume_update(
    payload: Any,
    collector: TraceCollector,
) -> tuple[list[dict] | None, str | None]:
    """RAG ``updates`` 流：累积 trace，并把 sources/retrieval_question 透传出去。"""
    if not isinstance(payload, dict):
        return None, None

    new_sources: list[dict] | None = None
    new_rq: str | None = None
    for _node_name, update in payload.items():
        if not isinstance(update, dict):
            continue
        collector.add_rag_entries(update.get("tool_trace"))
        if "sources" in update:
            new_sources = list(update["sources"] or [])
        if "retrieval_question" in update and isinstance(
            update["retrieval_question"], str
        ):
            new_rq = update["retrieval_question"]
    return new_sources, new_rq


def _consume_agent_update(
    payload: Any,
    collector: TraceCollector,
) -> tuple[list[dict] | None, str | None]:
    """Agent ``updates`` 流：把 messages 喂给 collector，提取 sources/retrieval_question。"""
    if not isinstance(payload, dict):
        return None, None

    new_sources: list[dict] | None = None
    new_rq: str | None = None
    for _node_name, update in payload.items():
        if not isinstance(update, dict):
            continue
        collector.add_agent_messages(update.get("messages"))
        if "sources" in update:
            sources_value = update["sources"]
            if isinstance(sources_value, list) and sources_value:
                new_sources = list(sources_value)
        if "retrieval_question" in update and isinstance(
            update["retrieval_question"], str
        ):
            new_rq = update["retrieval_question"]
    return new_sources, new_rq
