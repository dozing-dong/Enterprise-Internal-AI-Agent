"""Agent 主循环：plan -> execute -> observe，直到拿到最终回答。

两个入口：
- ``run``：非流式，LangGraph StateGraph 驱动，返回 ``AgentRunResult``，给 POST /chat 用。
- ``run_stream``：流式生成器，逐步 yield ``AgentEvent``，给 POST /chat/stream 用。

设计原则：
- RAG 是黑盒：通过 ``rag_answer`` 工具调用，runner 不直接组装 retriever。
- 模型自决：每一步由 Bedrock Converse 决定调用哪个工具或直接给答案。
- 失败降级：planner 异常或 max_steps 用尽，统一回退到 ``chat_executor``。
- 历史一致：用户问题 + 最终回答按 rag 模式同样的方式落库，前端体验一致。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from backend.agent.builtin_tools import ChatExecutor
from backend.agent.graph import build_agent_graph
from backend.agent.policy import AGENT_MAX_STEPS, AGENT_SYSTEM_PROMPT
from backend.agent.planner import plan_step_stream
from backend.agent.schemas import (
    AgentEvent,
    AgentRunResult,
    AgentStep,
    ToolCall,
)
from backend.agent.steps import (
    build_initial_messages,
    build_tool_result_message,
    summarize_tool_data,
)
from backend.agent.tools import ToolRegistry
from backend.storage.history import append_session_messages


class AgentRunner:
    """单 agent + 多工具的执行器。

    线程安全：方法本身无共享可变状态（每次调用独立构造 messages 与
    decision_trace），可以被多请求并发复用。底层 Bedrock client 以 lru_cache
    复用连接，本身线程安全。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        chat_executor_fallback: ChatExecutor,
        max_steps: int = AGENT_MAX_STEPS,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
    ) -> None:
        self._registry = registry
        self._chat_executor_fallback = chat_executor_fallback
        self._max_steps = max_steps
        self._system_prompt = system_prompt
        self._graph = build_agent_graph(
            registry,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    def _persist_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """把这一轮的用户问题与最终回答写入历史存储。"""
        if not answer:
            return
        append_session_messages(
            session_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )

    def _fallback_via_rag(
        self,
        question: str,
        session_id: str,
        *,
        reason: str,
        decision_trace: list[dict],
    ) -> AgentRunResult:
        """Agent 流程出错时，直接走一遍现有 RAG 流水线兜底。

        ``chat_executor`` 自己会把历史落库，因此这里不再 ``_persist_turn``。
        """
        result = self._chat_executor_fallback(question, session_id)
        decision_trace.append(
            {
                "index": len(decision_trace) + 1,
                "fallback": True,
                "reason": reason,
            }
        )
        return AgentRunResult(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            retrieval_question=result.get("retrieval_question", question),
            decision_trace=decision_trace,
            fallback=True,
        )

    def run(self, question: str, session_id: str) -> AgentRunResult:
        """同步执行 agent，直到拿到最终答案或耗尽预算。

        任何异常都会触发一次 fallback，对外保证“总能给出答复”。
        """
        decision_trace: list[dict] = []
        messages = build_initial_messages(question, session_id)

        try:
            out = self._graph.invoke(
                {
                    "question": question,
                    "session_id": session_id,
                    "messages": messages,
                    "decision_trace": [],
                    "sources": [],
                    "retrieval_question": None,
                    "step_index": 0,
                    "last_response": {},
                    "stop_reason": "",
                    "final_text": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - 故意吞下，统一兜底
            return self._fallback_via_rag(
                question,
                session_id,
                reason=f"{type(exc).__name__}: {exc}",
                decision_trace=decision_trace,
            )

        decision_trace = list(out.get("decision_trace", []) or [])
        sources = list(out.get("sources", []) or [])
        retrieval_question = out.get("retrieval_question")
        final_text = (out.get("final_text") or "").strip()

        if not final_text:
            last = decision_trace[-1] if decision_trace else {}
            reason = (
                "empty_final_text"
                if "final_answer" in last
                else "max_steps_exhausted"
            )
            return self._fallback_via_rag(
                question,
                session_id,
                reason=reason,
                decision_trace=decision_trace,
            )

        self._persist_turn(session_id, question, final_text)

        return AgentRunResult(
            answer=final_text,
            sources=sources,
            retrieval_question=retrieval_question,
            decision_trace=decision_trace,
            fallback=False,
        )

    def run_stream(
        self,
        question: str,
        session_id: str,
    ) -> Iterator[AgentEvent]:
        """流式执行 agent；逐步 yield 结构化事件给路由层。

        事件类型与含义：
        - ``progress``: ``{stage, message}`` 阶段提示，沿用 rag 模式同名事件。
        - ``tool_call``: ``{name, arguments, tool_use_id, step}``
        - ``tool_result``: ``{name, ok, error?, summary?, latency_ms}``
        - ``sources``: 仅当 ``rag_answer`` 工具命中后下发，schema 与 rag 模式一致。
        - ``token``: ``{text}`` 模型流式文本片段。
        - ``done``: ``{full_answer, decision_trace, fallback, ...}``
        - ``error``: ``{detail}`` 兜底失败时下发。
        """
        decision_trace: list[dict] = []
        messages = build_initial_messages(question, session_id)
        sources: list[dict] = []
        retrieval_question: str | None = None
        final_answer_chunks: list[str] = []

        try:
            yield AgentEvent(
                "progress",
                {"stage": "deciding", "message": "Thinking..."},
            )

            for step_index in range(1, self._max_steps + 1):
                step_text_chunks: list[str] = []
                step_tool_uses: list[dict[str, Any]] = []
                stop_reason = ""

                for event in plan_step_stream(
                    messages,
                    registry=self._registry,
                    system_prompt=self._system_prompt,
                ):
                    etype = event.get("type")
                    if etype == "text_delta":
                        text = event.get("text", "")
                        if text:
                            step_text_chunks.append(text)
                            final_answer_chunks.append(text)
                            yield AgentEvent("token", {"text": text})
                        continue

                    if etype == "tool_use":
                        step_tool_uses.append(
                            {
                                "tool_use_id": event.get("tool_use_id", ""),
                                "name": event.get("name", ""),
                                "input": event.get("input") or {},
                            }
                        )
                        continue

                    if etype == "stop":
                        stop_reason = event.get("stop_reason", "")
                        break

                if stop_reason == "tool_use" and step_tool_uses:
                    first_use = step_tool_uses[0]
                    call = ToolCall(
                        name=first_use["name"],
                        arguments=first_use.get("input") or {},
                        tool_use_id=first_use["tool_use_id"],
                    )

                    yield AgentEvent(
                        "progress",
                        {
                            "stage": "tool_running",
                            "message": f"Calling tool: {call.name}",
                        },
                    )
                    yield AgentEvent(
                        "tool_call",
                        {
                            "step": step_index,
                            "name": call.name,
                            "arguments": call.arguments,
                            "tool_use_id": call.tool_use_id,
                        },
                    )

                    result = self._registry.execute(
                        call,
                        context={"session_id": session_id},
                    )

                    yield AgentEvent(
                        "tool_result",
                        {
                            "step": step_index,
                            "name": call.name,
                            "ok": result.ok,
                            "error": result.error,
                            "latency_ms": result.latency_ms,
                            "summary": summarize_tool_data(result),
                        },
                    )

                    if result.ok and call.name == "rag_answer":
                        tool_sources = result.data.get("sources", []) or []
                        if tool_sources:
                            sources = tool_sources
                            retrieval_question = result.data.get(
                                "retrieval_question"
                            ) or retrieval_question
                            yield AgentEvent(
                                "sources",
                                {
                                    "sources": tool_sources,
                                    "retrieval_question": retrieval_question
                                    or question,
                                    "original_question": question,
                                },
                            )

                    decision_trace.append(
                        AgentStep(
                            index=step_index,
                            thought="".join(step_text_chunks) or None,
                            tool_call=call,
                            tool_result=result,
                        ).to_dict()
                    )

                    raw_content: list[dict[str, Any]] = []
                    joined_text = "".join(step_text_chunks)
                    if joined_text:
                        raw_content.append({"text": joined_text})
                    for tu in step_tool_uses:
                        raw_content.append(
                            {
                                "toolUse": {
                                    "toolUseId": tu["tool_use_id"],
                                    "name": tu["name"],
                                    "input": tu["input"],
                                }
                            }
                        )

                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append(build_tool_result_message(result))

                    yield AgentEvent(
                        "progress",
                        {
                            "stage": "deciding",
                            "message": "Reviewing tool result...",
                        },
                    )
                    continue

                decision_trace.append(
                    AgentStep(
                        index=step_index,
                        final_answer="".join(step_text_chunks),
                    ).to_dict()
                )
                break
            else:
                yield from self._stream_fallback(
                    question,
                    session_id,
                    reason="max_steps_exhausted",
                    decision_trace=decision_trace,
                    already_streamed=bool(final_answer_chunks),
                )
                return
        except Exception as exc:  # noqa: BLE001
            yield from self._stream_fallback(
                question,
                session_id,
                reason=f"{type(exc).__name__}: {exc}",
                decision_trace=decision_trace,
                already_streamed=bool(final_answer_chunks),
            )
            return

        full_answer = "".join(final_answer_chunks).strip()

        if not full_answer:
            yield from self._stream_fallback(
                question,
                session_id,
                reason="empty_final_text",
                decision_trace=decision_trace,
                already_streamed=False,
            )
            return

        self._persist_turn(session_id, question, full_answer)

        yield AgentEvent(
            "done",
            {
                "session_id": session_id,
                "full_answer": full_answer,
                "original_question": question,
                "retrieval_question": retrieval_question or question,
                "decision_trace": decision_trace,
                "fallback": False,
            },
        )

    def _stream_fallback(
        self,
        question: str,
        session_id: str,
        *,
        reason: str,
        decision_trace: list[dict],
        already_streamed: bool,
    ) -> Iterator[AgentEvent]:
        """流式 fallback：直接调一次 chat_executor，把结果一次性下发。

        ``already_streamed=True`` 时说明已经流过部分 token，
        为了不让前端拼接产生乱码，把 fallback 的 full_answer 整个交给 done 事件，
        前端依据 done.full_answer 覆盖最终内容（与现有行为一致）。
        """
        decision_trace.append(
            {
                "index": len(decision_trace) + 1,
                "fallback": True,
                "reason": reason,
            }
        )

        try:
            result = self._chat_executor_fallback(question, session_id)
        except Exception as exc:  # noqa: BLE001
            yield AgentEvent(
                "error",
                {"detail": f"agent fallback failed: {exc}"},
            )
            return

        fallback_sources = result.get("sources", []) or []
        retrieval_question = result.get("retrieval_question", question)
        answer = result.get("answer", "") or ""

        yield AgentEvent(
            "progress",
            {
                "stage": "generating",
                "message": "Falling back to RAG pipeline...",
            },
        )

        if fallback_sources:
            yield AgentEvent(
                "sources",
                {
                    "sources": fallback_sources,
                    "retrieval_question": retrieval_question,
                    "original_question": question,
                },
            )

        if not already_streamed and answer:
            yield AgentEvent("token", {"text": answer})

        yield AgentEvent(
            "done",
            {
                "session_id": session_id,
                "full_answer": answer,
                "original_question": question,
                "retrieval_question": retrieval_question,
                "decision_trace": decision_trace,
                "fallback": True,
            },
        )
