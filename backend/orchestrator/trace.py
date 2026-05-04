"""Unified call trace data structure and accumulator.

The two RAG and Agent modes have different sources:
- RAG: each LangGraph node (rewrite_query / vector_retrieve / ...)
  pushes incremental ``tool_trace`` entries; the accumulator
  normalizes each dict into a ``TraceStep``.
- Agent: from the ``messages`` updates emitted by the ToolNode, pull
  ``ToolMessage`` and the matching ``AIMessage.tool_calls`` and convert
  them into ``TraceStep`` entries.

In both cases the final output is a ``list[TraceStep]`` so the frontend
renders a single format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceStep:
    """A single call trace entry.

    Fields are kept lean-just enough for the frontend to render:
    - ``step``: step number (1-based).
    - ``name``: node / tool name.
    - ``input_summary`` / ``output_summary``: human-readable summaries (truncated).
    - ``ok``: whether it succeeded; on failure ``error`` carries the cause.
    - ``latency_ms``: optional, for observing performance hotspots.
    - ``agent``: in multi_agent mode, the sub-agent that produced this
      step (supervisor / policy / external_context / writer); ``None`` in other modes.
    """

    step: int
    name: str
    input_summary: str | None = None
    output_summary: str | None = None
    ok: bool = True
    latency_ms: int | None = None
    error: str | None = None
    agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate(value: Any, *, limit: int = 160) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    if not text:
        return None
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "\u2026"
    return text


def _summarize_rag_step(entry: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract unified input/output summaries from the tool_trace fields a RAG node writes."""
    name = entry.get("tool", "")

    input_summary: str | None = None
    output_summary: str | None = None

    if "input" in entry:
        input_summary = _truncate(entry["input"])
    elif "input_vector_count" in entry or "input_keyword_count" in entry:
        input_summary = (
            f"vector={entry.get('input_vector_count', 0)},"
            f" keyword={entry.get('input_keyword_count', 0)}"
        )
    elif "input_count" in entry:
        input_summary = f"candidates={entry['input_count']}"

    if name == "generate_answer":
        chars = entry.get("output_chars")
        if isinstance(chars, int):
            output_summary = f"{chars} chars"
    elif "output" in entry:
        output_summary = _truncate(entry["output"])
    elif "output_count" in entry:
        output_summary = f"{entry['output_count']} docs"

    return input_summary, output_summary


@dataclass
class TraceCollector:
    """Streamingly accumulate TraceSteps.

    Called by the orchestrator when node updates arrive on the
    ``updates`` stream.
    """

    steps: list[TraceStep] = field(default_factory=list)
    _next_index: int = 1

    def add_rag_entries(
        self,
        entries: list[dict[str, Any]] | None,
        *,
        agent: str | None = None,
    ) -> None:
        """RAG mode: append tool_trace entries newly emitted by a node to the trace.

        ``agent`` is optional: the PolicyAgent subgraph dispatched by
        multi_agent also pushes RAG-style tool_trace entries; in that case
        each step is tagged with the corresponding sub-agent.
        """
        if not entries:
            return
        for entry in entries:
            name = entry.get("tool")
            if not isinstance(name, str) or not name:
                continue
            input_summary, output_summary = _summarize_rag_step(entry)
            self.steps.append(
                TraceStep(
                    step=self._next_index,
                    name=name,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    agent=agent,
                )
            )
            self._next_index += 1

    def add_agent_messages(
        self,
        messages: list[Any] | None,
        *,
        agent: str | None = None,
    ) -> None:
        """Agent mode: pull tool calls and results from ToolNode/agent-node messages.

        - ``AIMessage.tool_calls``: tool calls requested by the model this turn -> staged
        - ``ToolMessage``: tool execution result -> paired with a previously-staged tool_call to form a step
        - ``agent``: optional, marks this batch of messages as belonging to a particular sub-agent (used by multi_agent).
        """
        if not messages:
            return

        # Pull staged-but-unmatched tool_calls from the existing steps.
        # Simple implementation: maintain a per-turn tool_use_id -> step-index map.
        for msg in messages:
            if _is_ai_message_with_tool_calls(msg):
                for call in msg.tool_calls:
                    name = call.get("name") or ""
                    args = call.get("args") or {}
                    self.steps.append(
                        TraceStep(
                            step=self._next_index,
                            name=name,
                            input_summary=_truncate(args),
                            output_summary=None,
                            agent=agent,
                        )
                    )
                    self._pending_call_index_by_id[call.get("id", "")] = (
                        len(self.steps) - 1
                    )
                    self._next_index += 1
            elif _is_tool_message(msg):
                tool_call_id = getattr(msg, "tool_call_id", "") or ""
                idx = self._pending_call_index_by_id.pop(tool_call_id, None)
                content = getattr(msg, "content", "")
                payload = _safe_parse_json(content)
                ok = True
                error: str | None = None
                if isinstance(payload, dict):
                    ok = bool(payload.get("ok", True))
                    if not ok:
                        error = str(payload.get("error", ""))
                output_summary = _truncate(payload if payload is not None else content)

                if idx is not None and 0 <= idx < len(self.steps):
                    prev = self.steps[idx]
                    self.steps[idx] = TraceStep(
                        step=prev.step,
                        name=prev.name,
                        input_summary=prev.input_summary,
                        output_summary=output_summary,
                        ok=ok,
                        error=error,
                        agent=prev.agent or agent,
                    )
                else:
                    name = getattr(msg, "name", "") or "tool"
                    self.steps.append(
                        TraceStep(
                            step=self._next_index,
                            name=name,
                            output_summary=output_summary,
                            ok=ok,
                            error=error,
                            agent=agent,
                        )
                    )
                    self._next_index += 1

    def add_node_step(
        self,
        name: str,
        *,
        agent: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
    ) -> None:
        """multi_agent mode: directly register a node-level step (not a tool call).

        For example the supervisor's decision or the writer's final
        generation. These are neither ToolMessages nor RAG-node
        tool_trace entries, so they go through this dedicated entry point.
        """
        if not name:
            return
        self.steps.append(
            TraceStep(
                step=self._next_index,
                name=name,
                input_summary=input_summary,
                output_summary=output_summary,
                agent=agent,
            )
        )
        self._next_index += 1

    def to_list(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    # Internal: tool_call_id -> step index, used to bind an
    # AIMessage.tool_calls entry with the ToolMessage result that arrives later.
    _pending_call_index_by_id: dict[str, int] = field(default_factory=dict)


def _is_ai_message_with_tool_calls(msg: Any) -> bool:
    tool_calls = getattr(msg, "tool_calls", None)
    return bool(tool_calls)


def _is_tool_message(msg: Any) -> bool:
    return msg.__class__.__name__ == "ToolMessage"


def _safe_parse_json(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
