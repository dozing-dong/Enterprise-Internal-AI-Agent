"""AgentRunner：薄包装器，对外暴露异步流式接口。

orchestrator 通常直接消费 ``runtime.agent_graph.astream``，这里保留一个
``AgentRunner`` 主要是为：
- 让 ``runtime`` 与 ``orchestrator`` 之间通过明确的 facade 解耦；
- 在未来想加 per-run 上下文（trace id 之类）时有插点。

不再有非流式 ``run`` 与 RAG fallback。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from backend.agent.graph import build_initial_messages
from backend.storage.history import read_session_history


class AgentRunner:
    """编译好的 Agent LangGraph 的薄包装。"""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @property
    def graph(self) -> Any:
        return self._graph

    def _initial_state(self, question: str, session_id: str) -> dict[str, Any]:
        history = read_session_history(session_id)
        return {
            "messages": build_initial_messages(history, question),
            "session_id": session_id,
            "sources": [],
            "retrieval_question": None,
            "original_question": question,
        }

    async def astream(
        self,
        question: str,
        session_id: str,
        *,
        stream_mode: list[str] | None = None,
    ) -> AsyncIterator[Any]:
        modes = stream_mode or ["messages", "updates"]
        async for chunk in self._graph.astream(
            self._initial_state(question, session_id),
            stream_mode=modes,
        ):
            yield chunk

    def stream(
        self,
        question: str,
        session_id: str,
        *,
        stream_mode: list[str] | None = None,
    ) -> Iterator[Any]:
        modes = stream_mode or ["messages", "updates"]
        for chunk in self._graph.stream(
            self._initial_state(question, session_id),
            stream_mode=modes,
        ):
            yield chunk
