from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple

from backend.api.exceptions import RagException
from backend.api.schemas import ChatRequest, ChatResponse, ChatStreamRequest, SourceItem
from backend.rag.title import generate_session_title
from backend.runtime import DemoRuntime
from backend.storage.history import build_history_path, read_session_history
from backend.storage.sessions import rename_session, touch_session


class OrchestratorStreamEvent(NamedTuple):
    type: str
    data: dict[str, Any]


class ChatOrchestrator:
    """Controller 下层编排：mode 分支、流式 RAG/Agent、首回合标题与 done 合并。"""

    def __init__(self, runtime: DemoRuntime) -> None:
        self._runtime = runtime

    def run(self, request: ChatRequest) -> ChatResponse:
        if request.mode == "agent":
            return self._run_agent(request)
        try:
            result = self._runtime.chat_executor(
                request.question,
                request.session_id,
            )
        except RuntimeError as exc:
            raise RagException(str(exc), status_code=503) from exc
        except Exception as exc:
            raise RagException(
                f"RAG 执行时发生内部错误：{exc}", status_code=500
            ) from exc

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

    def _run_agent(self, request: ChatRequest) -> ChatResponse:
        if self._runtime.agent_runner is None:
            raise RagException("Agent runner 未初始化。", status_code=503)

        try:
            run_result = self._runtime.agent_runner.run(
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
            retrieval_question=run_result.retrieval_question
            or request.question,
            session_id=request.session_id,
            history_file=str(history_file),
            sources=[SourceItem(**s) for s in run_result.sources],
            decision_trace=run_result.decision_trace,
            fallback=run_result.fallback,
            mode="agent",
        )

    def stream(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
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

    def _stream_rag(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
        rag_service = self._runtime.rag_service

        existing_history = read_session_history(request.session_id)
        token_parts: list[str] = []
        retrieval_question = request.question

        try:
            for event_type, data in rag_service.stream(
                request.question,
                request.session_id,
                existing_history=existing_history,
            ):
                yield OrchestratorStreamEvent(event_type, data)
                if event_type == "token":
                    token_parts.append(data.get("text", ""))
                elif event_type == "sources":
                    retrieval_question = data.get(
                        "retrieval_question", retrieval_question
                    )
        except Exception as exc:  # noqa: BLE001
            yield OrchestratorStreamEvent("error", {"detail": str(exc)})
            return

        full_answer = "".join(token_parts).strip() or "No answer generated."

        touch_session(request.session_id)

        title = session_record_title
        if is_first_turn and full_answer:
            yield OrchestratorStreamEvent(
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
                pass

        yield OrchestratorStreamEvent(
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

    def _stream_agent(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
        agent_runner = self._runtime.agent_runner

        try:
            done_payload: dict[str, Any] | None = None

            for event in agent_runner.run_stream(
                request.question,
                request.session_id,
            ):
                if event.type == "done":
                    done_payload = dict(event.data)
                    continue

                yield OrchestratorStreamEvent(event.type, event.data)

            touch_session(request.session_id)

            full_answer = (done_payload or {}).get("full_answer", "")
            title = session_record_title
            if is_first_turn and full_answer:
                yield OrchestratorStreamEvent(
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
            yield OrchestratorStreamEvent("done", final_done)
        except Exception as exc:  # noqa: BLE001
            yield OrchestratorStreamEvent("error", {"detail": str(exc)})
