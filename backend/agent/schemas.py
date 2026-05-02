"""Agent 层使用的数据结构。

刻意只用 dataclass + 简单 dict，避免引入 pydantic 依赖；
对外通过 ``to_dict`` 序列化，便于直接写入 SSE / JSON 响应。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ToolCall:
    """一次工具调用请求。

    ``tool_use_id`` 由 LLM（Bedrock toolUse 块）分配，agent 在回填
    toolResult 时必须原样带回，模型才能把结果对到当初的调用上。
    """

    name: str
    arguments: dict[str, Any]
    tool_use_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "tool_use_id": self.tool_use_id,
        }


@dataclass(slots=True)
class ToolResult:
    """工具执行结果。

    保持 ``ok / data / error`` 三段式，方便 planner 在下一步决策时
    通过 ``ok`` 字段直接判断是否需要重试或换工具。
    """

    tool_use_id: str
    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_use_id": self.tool_use_id,
            "name": self.name,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }

    def to_observation_text(self) -> str:
        """把工具结果序列化成喂给模型的文本观察。

        Bedrock toolResult 块允许直接传 JSON 字符串，
        这里把整个对象序列化以保留结构信息（来源、错误、片段计数等）。
        """
        import json

        return json.dumps(
            {
                "ok": self.ok,
                "data": self.data,
                "error": self.error,
            },
            ensure_ascii=False,
        )


@dataclass(slots=True)
class AgentStep:
    """决策循环里的一步快照，主要用于可观测性。"""

    index: int
    thought: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    final_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "final_answer": self.final_answer,
        }


@dataclass(slots=True)
class AgentRunResult:
    """非流式运行的最终结果。

    ``decision_trace`` 是给 API/前端用的“决策日志”，每一项就是一个 step；
    ``sources`` 与 ``retrieval_question`` 仅当本轮用到 ``rag_answer`` 时填充。
    """

    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_question: str | None = None
    decision_trace: list[dict] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_question": self.retrieval_question,
            "decision_trace": self.decision_trace,
            "fallback": self.fallback,
        }


# 流式事件的判别字段；与前端 sse.ts 中的 event 名一一对应。
AgentEventType = Literal[
    "progress",
    "tool_call",
    "tool_result",
    "sources",
    "token",
    "done",
    "error",
]


@dataclass(slots=True)
class AgentEvent:
    """流式事件的统一封装。

    路由层把它直接映射到 SSE 帧：``event: <type>\\ndata: <json>``。
    """

    type: AgentEventType
    data: dict[str, Any]
