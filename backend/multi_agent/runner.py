"""MultiAgentRunner: thin wrapper providing sync / async streaming for the orchestrator.

Mirrors [backend/agent/runner.py](backend/agent/runner.py:1) so the
orchestrator can consume both graphs through the same protocol
(``stream`` + ``astream``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from backend.multi_agent.state import build_initial_multi_agent_state
from backend.storage.history import read_session_history


class MultiAgentRunner:
    """Thin wrapper over a compiled multi-agent LangGraph."""

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
