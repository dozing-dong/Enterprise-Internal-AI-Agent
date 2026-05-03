"""MultiAgentRunner：薄包装器，给 orchestrator 提供同步 / 异步流式接口。

形态与 [backend/agent/runner.py](backend/agent/runner.py:1) 保持一致，便于
orchestrator 通过同样的协议消费两套图（``stream`` + ``astream``）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from backend.multi_agent.state import build_initial_multi_agent_state
from backend.storage.history import read_session_history


class MultiAgentRunner:
    """编译好的多 Agent LangGraph 的薄包装。"""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @property
    def graph(self) -> Any:
        return self._graph

    def _initial_state(self, question: str, session_id: str) -> dict[str, Any]:
        history = read_session_history(session_id)
        return build_initial_multi_agent_state(
            question=question,
            session_id=session_id,
            history=history,
        )

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
