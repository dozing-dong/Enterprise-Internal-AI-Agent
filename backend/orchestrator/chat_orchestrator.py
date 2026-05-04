"""Lower-level controller orchestration: a single stream() + aggregate(), all LangGraph and all streaming.

Design notes:
- ``stream(...)``: select the RAG / Agent compiled graph by mode and
  uniformly drive ``graph.stream(stream_mode=["messages","updates"])``:
  - ``messages`` stream: forwards only text chunks from "user-visible"
    model nodes (filtered by tag) so RAG's internal ``generate_answer``
    is not double-emitted when the Agent calls the RAG tool.
  - ``updates`` stream: accumulates the unified trace via
    ``TraceCollector``; whenever a node writes ``sources`` /
    ``retrieval_question``, immediately emits one ``sources`` event.
- At the end, runs ``touch_session`` / first-turn title generation /
  emits ``done``.
- ``aggregate(...)``: used by ``POST /chat``; consumes ``stream``
  internally, joins tokens into a complete ``answer``, and returns
  ``ChatResponse``.
"""

from __future__ import annotations

import json
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
from backend.multi_agent import (
    AGENT_NAME_EXTERNAL,
    AGENT_NAME_POLICY,
    AGENT_NAME_SUPERVISOR,
    AGENT_NAME_WRITER,
    WRITER_TAG,
    build_initial_multi_agent_state,
)
from backend.orchestrator.trace import TraceCollector
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


# Name of the "user-visible" final generation node in RAG. The orchestrator
# only forwards tokens from this node, so when an Agent calls the RAG tool,
# RAG's internal generation is not also pushed to the frontend.
_RAG_FINAL_NODE_NAME = "generate_answer"

# In the multi_agent topology: parent-graph node name -> the sub-agent it
# belongs to, so messages / tool_trace produced inside a node can be tagged
# with the corresponding ``trace.agent`` field.
_MULTI_AGENT_NODE_TO_AGENT: dict[str, str] = {
    "supervisor": AGENT_NAME_SUPERVISOR,
    "policy": AGENT_NAME_POLICY,
    "external": AGENT_NAME_EXTERNAL,
    "writer": AGENT_NAME_WRITER,
}


class OrchestratorStreamEvent(NamedTuple):
    type: str
    data: dict[str, Any]


class ChatOrchestrator:
    """Unified streaming entry point for the RAG / Agent modes."""

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
        """Pick the graph by mode and drive streaming output."""
        if request.mode == "agent":
            yield from self._stream_agent(
                request,
                session_record_title=session_record_title,
                is_first_turn=is_first_turn,
            )
        elif request.mode == "multi_agent":
            yield from self._stream_multi_agent(
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
        """Non-streaming entry: consumes a stream once and aggregates the result into a ChatResponse."""
        stream_request = ChatStreamRequest(
            question=request.question,
            session_id=request.session_id,
            mode=request.mode,
        )
        # Session metadata / first-turn detection are needed by the stream
        # entry point too; prepare them uniformly here.
        session_record = create_session_if_missing(request.session_id)
        existing_history = read_session_history(request.session_id)
        is_first_turn = len(existing_history) == 0

        token_parts: list[str] = []
        sources: list[dict] = []
        retrieval_question = request.question
        trace: list[dict] = []
        agents_invoked: list[str] = []
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
                agents_invoked = event.data.get("agents_invoked", []) or []
                title = event.data.get("title", title)
                full_answer = event.data.get("full_answer", "")
                if full_answer and not token_parts:
                    token_parts = [full_answer]
            elif event.type == "error":
                error = event.data.get("detail", "")
                break

        if error:
            raise RagException(
                f"Internal error during {request.mode.upper()} execution: {error}",
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
            agents_invoked=agents_invoked,
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
                "error", {"detail": "RAG graph is not initialized."}
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
                "error", {"detail": "Agent graph is not initialized."}
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

    # ---- Multi-Agent mode ------------------------------------------------

    def _stream_multi_agent(
        self,
        request: ChatStreamRequest,
        *,
        session_record_title: str,
        is_first_turn: bool,
    ) -> Iterator[OrchestratorStreamEvent]:
        graph = self._runtime.multi_agent_graph
        if graph is None:
            yield OrchestratorStreamEvent(
                "error", {"detail": "Multi-Agent graph is not initialized."}
            )
            return

        collector = TraceCollector()
        token_parts: list[str] = []
        sources: list[dict] = []
        agents_invoked: list[str] = []
        last_sources_signature: int | None = None
        # The Writer writes the final answer back via state.final_answer;
        # when the messages stream produces no chunks (e.g. a fake-model
        # test or the model did not actually stream), use this fallback as full_answer.
        final_answer_from_state: str = ""

        history = read_session_history(request.session_id)
        initial_state = build_initial_multi_agent_state(
            question=request.question,
            session_id=request.session_id,
            history=history,
        )

        try:
            stream_iter = graph.stream(
                initial_state,
                stream_mode=["messages", "updates"],
            )
            for stream_mode, payload in stream_iter:
                if stream_mode == "messages":
                    text = _extract_user_visible_text(
                        payload,
                        allowed_tag=WRITER_TAG,
                    )
                    if text:
                        token_parts.append(text)
                        yield OrchestratorStreamEvent("token", {"text": text})
                elif stream_mode == "updates":
                    (
                        new_sources,
                        new_invoked,
                        new_final_answer,
                    ) = _consume_multi_agent_update(payload, collector)
                    if new_sources is not None:
                        # Accumulate sources across all sub-agent updates so
                        # that employee-directory results (written by supervisor)
                        # are not overwritten when policy later writes RAG chunks.
                        sources = _merge_sources(sources, new_sources)
                    if new_invoked:
                        for name in new_invoked:
                            if name and name not in agents_invoked:
                                agents_invoked.append(name)
                    if new_final_answer:
                        final_answer_from_state = new_final_answer
                    if (
                        new_sources is not None
                        and id(sources) != last_sources_signature
                    ):
                        last_sources_signature = id(sources)
                        yield OrchestratorStreamEvent(
                            "sources",
                            {
                                "sources": sources,
                                "retrieval_question": request.question,
                                "original_question": request.question,
                            },
                        )
        except Exception as exc:  # noqa: BLE001
            yield OrchestratorStreamEvent("error", {"detail": str(exc)})
            return

        full_answer = "".join(token_parts).strip()
        if not full_answer and final_answer_from_state:
            full_answer = final_answer_from_state.strip()

        if full_answer:
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
                "retrieval_question": request.question,
                "trace": collector.to_list(),
                "mode": "multi_agent",
                "agents_invoked": agents_invoked,
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
    """Extract visible text from a chunk on LangGraph's ``messages`` stream.

    ``payload`` is a ``(message_chunk, metadata)`` tuple. Filter by
    ``allowed_node`` or ``allowed_tag`` to avoid forwarding the internal
    generation of subgraphs (e.g. RAG-as-tool) to the frontend.
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
                # ChatBedrockConverse streaming chunks look like {"type":"text","text":"..."}.
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
    """RAG ``updates`` stream: accumulate trace and pass through sources/retrieval_question."""
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
    """Agent ``updates`` stream: feed messages to the collector and extract sources/retrieval_question."""
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


def _consume_multi_agent_update(
    payload: Any,
    collector: TraceCollector,
) -> tuple[list[dict] | None, list[str], str | None]:
    """Multi-Agent ``updates`` stream:

    - Extract ``agents_invoked`` / ``sources`` for each sub-agent node;
    - Push a node-level step (with ``agent`` tag) to the trace collector;
    - Extract the ``final_answer`` written back by the writer node, used
      as a fallback when the token stream is empty.
    - Does NOT extract ``retrieval_question``: multi_agent does not expose
      RAG's rewritten query.
    """
    if not isinstance(payload, dict):
        return None, [], None

    new_sources: list[dict] | None = None
    new_invoked: list[str] = []
    new_final_answer: str | None = None

    for node_name, update in payload.items():
        if not isinstance(update, dict):
            continue

        sub_agent = _MULTI_AGENT_NODE_TO_AGENT.get(node_name)

        invoked = update.get("agents_invoked")
        if isinstance(invoked, list):
            for name in invoked:
                if isinstance(name, str) and name:
                    new_invoked.append(name)

        if "sources" in update:
            sources_value = update["sources"]
            if isinstance(sources_value, list) and sources_value:
                new_sources = list(sources_value)

        if (
            "final_answer" in update
            and isinstance(update["final_answer"], str)
            and update["final_answer"].strip()
        ):
            new_final_answer = update["final_answer"]

        # Expand the sub-agent's internal ReAct tool calls into trace
        # steps, each tagged with ``agent`` so the frontend can display
        # them grouped by sub-agent. The subgraph itself only exposes
        # policy_result / external_result on the parent updates stream
        # (no inner messages), so we derive from result.tool_calls.
        for inner_calls in _iter_inner_tool_calls(update):
            _record_tool_calls_to_trace(collector, inner_calls, agent=sub_agent)

        summary = _summarize_multi_agent_update(node_name, update)
        if sub_agent is not None and summary is not None:
            collector.add_node_step(
                name=node_name,
                agent=sub_agent,
                output_summary=summary,
            )

    return new_sources, new_invoked, new_final_answer


def _iter_inner_tool_calls(update: dict) -> Iterator[list[dict]]:
    """Pull a sub-agent's self-reported tool_calls list out of a parent-graph node update.

    ``policy_result.tool_calls`` and ``external_result.tool_calls`` use
    the same dict shape: ``{name, args, id, ok, result, error?}``.
    """
    for key in ("policy_result", "external_result"):
        result = update.get(key)
        if not isinstance(result, dict):
            continue
        calls = result.get("tool_calls")
        if isinstance(calls, list) and calls:
            yield calls


def _record_tool_calls_to_trace(
    collector: TraceCollector,
    calls: list[dict],
    *,
    agent: str | None,
) -> None:
    """Register each entry in a sub-agent's tool_calls list as a trace step."""
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        if not name:
            continue
        args = call.get("args") or {}
        ok = bool(call.get("ok", True))
        error = call.get("error") if not ok else None
        result = call.get("result")
        output_summary: str | None
        if result is None:
            output_summary = None
        else:
            try:
                output_summary = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError):
                output_summary = str(result)
            if output_summary and len(output_summary) > 160:
                output_summary = output_summary[:159].rstrip() + "\u2026"

        try:
            input_summary: str | None = (
                json.dumps(args, ensure_ascii=False) if args else None
            )
        except (TypeError, ValueError):
            input_summary = str(args) if args else None
        if input_summary and len(input_summary) > 160:
            input_summary = input_summary[:159].rstrip() + "\u2026"

        # Use the low-level add_node_step path: name=tool name, agent=owning sub-agent.
        collector.add_node_step(
            name=name,
            agent=agent,
            input_summary=input_summary,
            output_summary=output_summary,
        )
        # ok=False tool call: mark the last step as failed.
        if not ok and collector.steps:
            last = collector.steps[-1]
            last.ok = False
            last.error = str(error) if error else "tool reported failure"


def _merge_sources(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Accumulate sources from multiple sub-agents without dropping earlier results.

    Uses (context_id, content_prefix) as dedup key so that employee-directory
    entries written by the supervisor are preserved when the policy node later
    appends RAG chunks to the same sources channel.
    """
    seen: set[tuple] = set()
    merged: list[dict] = []
    for item in existing:
        if not isinstance(item, dict):
            continue
        key = _source_dedup_key(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    for item in incoming:
        if not isinstance(item, dict):
            continue
        key = _source_dedup_key(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _source_dedup_key(item: dict) -> tuple:
    metadata = item.get("metadata") or {}
    context_id = str(metadata.get("context_id", "")) if isinstance(metadata, dict) else ""
    content = str(item.get("content", ""))[:120]
    return (context_id, content)


def _summarize_multi_agent_update(node_name: str, update: dict) -> str | None:
    """Produce a brief output summary for each parent-graph node in multi_agent mode."""
    if node_name == "supervisor":
        plan = update.get("plan")
        if plan is None:
            return None
        try:
            return (
                f"use_policy={getattr(plan, 'use_policy', False)},"
                f" use_external={getattr(plan, 'use_external', False)},"
                f" locations={getattr(plan, 'locations', [])}"
            )
        except Exception:  # noqa: BLE001
            return None

    if node_name == "policy":
        result = update.get("policy_result") or {}
        if not isinstance(result, dict):
            return None
        if not result.get("ok", True):
            return f"failed: {result.get('error', 'unknown')}"
        answer = (result.get("answer") or "").strip()
        sources_count = len(result.get("sources") or [])
        return (
            f"summary_chars={len(answer)}, sources={sources_count}"
            if answer or sources_count
            else "no policy summary"
        )

    if node_name == "external":
        result = update.get("external_result") or {}
        if not isinstance(result, dict):
            return None
        if not result.get("ok", True):
            return f"failed: {result.get('error', 'unknown')}"
        tools_used = result.get("tools_used") or []
        return f"tools_used={list(tools_used)}"

    if node_name == "writer":
        answer = update.get("final_answer") or ""
        return f"answer_chars={len(answer)}"

    return None
