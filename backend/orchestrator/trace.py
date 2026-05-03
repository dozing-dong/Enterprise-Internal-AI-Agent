"""统一调用轨迹（Trace）数据结构与累积器。

RAG 与 Agent 两种模式来源不同：
- RAG：每个 LangGraph 节点（rewrite_query / vector_retrieve / ...）通过
  ``tool_trace`` 增量推送，累积器把每条 dict 标准化成 ``TraceStep``。
- Agent：从 ToolNode 推送的 ``messages`` 更新里抓 ``ToolMessage`` 与
  对应 ``AIMessage.tool_calls``，转换成 ``TraceStep``。

最终都吐成 ``list[TraceStep]``，前端只渲染一种格式。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceStep:
    """单条调用轨迹。

    字段保持精简，刚好够前端渲染：
    - ``step``：步骤编号（从 1 起）。
    - ``name``：节点 / 工具名。
    - ``input_summary`` / ``output_summary``：人可读的简述（已截断）。
    - ``ok``：是否成功；失败时 ``error`` 给原因。
    - ``latency_ms``：可选，便于观察性能瓶颈。
    """

    step: int
    name: str
    input_summary: str | None = None
    output_summary: str | None = None
    ok: bool = True
    latency_ms: int | None = None
    error: str | None = None

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
        return text[: limit - 1].rstrip() + "…"
    return text


def _summarize_rag_step(entry: dict[str, Any]) -> tuple[str | None, str | None]:
    """根据 RAG 节点写入的 tool_trace 字段，提取统一的 input/output 摘要。"""
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
    """流式累积 TraceStep。

    供 orchestrator 在 ``updates`` 流里 push 节点更新时调用。
    """

    steps: list[TraceStep] = field(default_factory=list)
    _next_index: int = 1

    def add_rag_entries(self, entries: list[dict[str, Any]] | None) -> None:
        """RAG 模式：把节点新增的 tool_trace 列表追加到轨迹。"""
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
                )
            )
            self._next_index += 1

    def add_agent_messages(self, messages: list[Any] | None) -> None:
        """Agent 模式：从 ToolNode/agent 节点产生的消息里抽工具调用与结果。

        - ``AIMessage.tool_calls``：模型本轮请求的工具调用 → 暂存
        - ``ToolMessage``：工具执行结果 → 与之前的 tool_call 配对成 step
        """
        if not messages:
            return

        # 先把已暂存但还没匹配的 tool_calls 从已有 step 中拿出来
        # 简单实现：维护一个本轮内的 tool_use_id -> step 索引映射
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
                    self.steps[idx] = TraceStep(
                        step=self.steps[idx].step,
                        name=self.steps[idx].name,
                        input_summary=self.steps[idx].input_summary,
                        output_summary=output_summary,
                        ok=ok,
                        error=error,
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
                        )
                    )
                    self._next_index += 1

    def to_list(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]

    # 内部：tool_call_id -> step 索引，用于把 AIMessage.tool_calls 与
    # 后到的 ToolMessage 结果绑定。
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
