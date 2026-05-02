"""内置工具实现。

- ``RagAnswerTool``：把现有 RAG 流水线作为黑盒，封装为一次性工具调用。
  这是最重要的工具：知识问答类问题首选它。
- ``CurrentTimeTool``：极简示例工具，主要用于验证多工具路由能力。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.agent.schemas import ToolResult


# 兼容现有 runtime.chat_executor 的签名：(question, session_id) -> dict
ChatExecutor = Callable[[str, str], dict[str, Any]]


class RagAnswerTool:
    """RAG 黑盒工具：一次调用 = 一次完整 RAG 流水线。

    内部直接复用 ``runtime.chat_executor``（即 langgraph 编译出的图），
    不重新拼接 retriever / reranker / generator，确保 agent 模式与 rag 模式
    完全等价的检索与生成行为，避免“两套实现走偏”。
    """

    name = "rag_answer"
    description = (
        "Answer a user question using the enterprise knowledge base. "
        "This tool runs the full retrieval-augmented-generation pipeline "
        "(query rewrite, hybrid retrieval, reranking, grounded generation) "
        "and returns a final answer along with the cited source snippets. "
        "Use this tool for any question that may require company knowledge, "
        "policy details, internal documents, or factual lookup. "
        "Prefer this tool over answering from your own memory."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The user's question to answer using the knowledge base. "
                    "Pass the original question verbatim; the tool will "
                    "perform its own internal query rewrite."
                ),
            }
        },
        "required": ["question"],
    }

    def __init__(self, chat_executor: ChatExecutor) -> None:
        self._chat_executor = chat_executor

    def invoke(
        self,
        arguments: dict[str, Any],
        *,
        context: dict[str, Any],
        tool_use_id: str,
    ) -> ToolResult:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return ToolResult(
                tool_use_id=tool_use_id,
                name=self.name,
                ok=False,
                error="argument 'question' must be a non-empty string",
            )

        session_id = context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return ToolResult(
                tool_use_id=tool_use_id,
                name=self.name,
                ok=False,
                error="missing session_id in agent context",
            )

        result = self._chat_executor(question, session_id)

        return ToolResult(
            tool_use_id=tool_use_id,
            name=self.name,
            ok=True,
            data={
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "retrieval_question": result.get(
                    "retrieval_question", question
                ),
                "tool_trace": result.get("tool_trace", []),
            },
        )


class CurrentTimeTool:
    """返回当前时间。

    存在的意义：验证“非知识类问题不强行走 RAG”，确保多工具路由有效。
    """

    name = "current_time"
    description = (
        "Return the current date and time. "
        "Use this when the user explicitly asks what time it is, "
        "what today's date is, or needs the current timestamp for something. "
        "Do NOT use this for questions that are about knowledge in documents."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "Optional IANA timezone name, e.g. 'Asia/Shanghai' "
                    "or 'UTC'. Defaults to UTC when omitted."
                ),
            }
        },
        "required": [],
    }

    def invoke(
        self,
        arguments: dict[str, Any],
        *,
        context: dict[str, Any],
        tool_use_id: str,
    ) -> ToolResult:
        tz_name = arguments.get("timezone")
        try:
            tz = ZoneInfo(tz_name) if isinstance(tz_name, str) and tz_name else timezone.utc
        except ZoneInfoNotFoundError:
            return ToolResult(
                tool_use_id=tool_use_id,
                name=self.name,
                ok=False,
                error=f"unknown timezone: {tz_name}",
            )

        now = datetime.now(tz)
        return ToolResult(
            tool_use_id=tool_use_id,
            name=self.name,
            ok=True,
            data={
                "iso": now.isoformat(),
                "timezone": str(tz),
                "epoch_seconds": int(now.timestamp()),
            },
        )
