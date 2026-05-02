"""Agent 每步共享的纯函数，供 LangGraph 节点与流式 runner 复用。"""

from __future__ import annotations

from typing import Any

from backend.agent.schemas import ToolResult
from backend.storage.history import read_session_history


def tool_result_status(ok: bool) -> str:
    """Bedrock 期望 toolResult.status 在 success / error 二选一。"""
    return "success" if ok else "error"


def build_tool_result_message(result: ToolResult) -> dict[str, Any]:
    """把 ToolResult 包装成可塞进 Converse messages 的 user 消息。"""
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": result.tool_use_id,
                    "content": [{"text": result.to_observation_text()}],
                    "status": tool_result_status(result.ok),
                }
            }
        ],
    }


def build_initial_messages(question: str, session_id: str) -> list[dict[str, Any]]:
    """加载历史 + 当前问题，得到本次调用的初始消息列表。"""
    history = read_session_history(session_id)
    messages: list[dict[str, Any]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def summarize_tool_data(result: ToolResult) -> dict[str, Any]:
    """提取 tool 结果中关键字段下发给前端，避免一次性塞太多文本。"""
    if not result.ok:
        return {"error": result.error}
    if result.name == "rag_answer":
        sources = result.data.get("sources", []) or []
        return {
            "sources_count": len(sources),
            "retrieval_question": result.data.get("retrieval_question"),
        }
    if result.name == "current_time":
        return {
            "iso": result.data.get("iso"),
            "timezone": result.data.get("timezone"),
        }
    return {}
