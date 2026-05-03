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


# RAG 节点中 "用户可见" 的最终生成节点名。orchestrator 只把该节点的
# token 转给前端，避免 Agent 调 RAG 工具时把 RAG 内部生成也下发。
_RAG_FINAL_NODE_NAME = "generate_answer"

# multi_agent 拓扑里：父图节点名 → 它属于哪个 sub-agent，便于把节点
# 内部产生的 messages / tool_trace 标记成对应的 trace.agent 字段。
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
                "error", {"detail": "Multi-Agent graph 未初始化。"}
            )
            return

        collector = TraceCollector()
        token_parts: list[str] = []
        sources: list[dict] = []
        agents_invoked: list[str] = []
        last_sources_signature: int | None = None
        # Writer 通过 state.final_answer 写回最终答复；当 messages 流没产生
        # 任何 chunk（例如 fake model 测试 / 模型未真正流式）时，用这个
        # fallback 作为 full_answer。
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
                        sources = new_sources
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


def _consume_multi_agent_update(
    payload: Any,
    collector: TraceCollector,
) -> tuple[list[dict] | None, list[str], str | None]:
    """Multi-Agent ``updates`` 流：

    - 把每个 sub-agent 节点的 ``agents_invoked`` / ``sources`` 提取出来；
    - 给 trace collector 推一条 node 级 step（带 ``agent`` 标记）；
    - 把 writer 节点写回的 ``final_answer`` 抽出，作为 token 流为空时的兜底。
    - 不提取 ``retrieval_question``：multi_agent 不暴露 RAG 改写后的 query。
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

        # 把 sub-agent 内部 ReAct 的 tool 调用展开成 trace step，每条带
        # 上 ``agent`` 标记，方便前端按 sub-agent 维度展示。子图本身在父
        # 图 updates 里只暴露 policy_result / external_result，不会暴露
        # inner messages 列表，所以这里从 result.tool_calls 派生。
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
    """从父图节点 update 里抽出 sub-agent 自报的 tool_calls 列表。

    ``policy_result.tool_calls`` 与 ``external_result.tool_calls`` 都用同一
    种 dict 形态：``{name, args, id, ok, result, error?}``。
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
    """把 sub-agent 暴露的 tool_calls 列表挨个登记成 trace step。"""
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
                output_summary = output_summary[:159].rstrip() + "…"

        try:
            input_summary: str | None = (
                json.dumps(args, ensure_ascii=False) if args else None
            )
        except (TypeError, ValueError):
            input_summary = str(args) if args else None
        if input_summary and len(input_summary) > 160:
            input_summary = input_summary[:159].rstrip() + "…"

        # 直接走低阶 add_node_step 简化路径：name=tool 名，agent=归属 sub-agent。
        collector.add_node_step(
            name=name,
            agent=agent,
            input_summary=input_summary,
            output_summary=output_summary,
        )
        # ok=False 的工具调用：把最后一个 step 标记为失败。
        if not ok and collector.steps:
            last = collector.steps[-1]
            last.ok = False
            last.error = str(error) if error else "tool reported failure"


def _summarize_multi_agent_update(node_name: str, update: dict) -> str | None:
    """为 multi_agent 的每个父图节点产出一条简短输出摘要。"""
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
