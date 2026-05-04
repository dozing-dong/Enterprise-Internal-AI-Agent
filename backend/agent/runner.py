"""AgentRunner: a thin wrapper exposing an async streaming interface.

The orchestrator usually consumes ``runtime.agent_graph.astream`` directly;
``AgentRunner`` is kept mainly to:
- decouple ``runtime`` and ``orchestrator`` via an explicit facade;
- provide an extension point for future per-run context (e.g. trace id).

There is no longer a non-streaming ``run`` or RAG fallback.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from backend.agent.graph import build_initial_messages
from backend.storage.history import read_session_history


class AgentRunner:
    """Thin wrapper over a compiled agent LangGraph."""

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
