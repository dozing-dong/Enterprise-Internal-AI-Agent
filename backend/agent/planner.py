"""Planner：把 Bedrock Converse 的工具决策能力封装成轻量函数。

非流式与流式各有一个入口，对应 ``AgentRunner.run`` 与 ``run_stream``。
两者共享同一份消息组装逻辑（在 ``backend/rag/models.py`` 中）。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from backend.agent.policy import AGENT_PLANNER_TEMPERATURE
from backend.agent.tools import ToolRegistry
from backend.rag.models import (
    chat_completion_with_tools,
    chat_completion_with_tools_stream,
)


def plan_step(
    messages: Sequence[dict[str, Any]],
    *,
    registry: ToolRegistry,
    system_prompt: str,
    temperature: float = AGENT_PLANNER_TEMPERATURE,
) -> dict[str, Any]:
    """非流式决策：调用一次 Converse，让模型决定是用工具还是直接回答。

    返回值字段含义见 ``chat_completion_with_tools``。
    """
    return chat_completion_with_tools(
        messages,
        tool_config=registry.to_bedrock_tool_config(),
        system_prompt=system_prompt,
        temperature=temperature,
    )


def plan_step_stream(
    messages: Sequence[dict[str, Any]],
    *,
    registry: ToolRegistry,
    system_prompt: str,
    temperature: float = AGENT_PLANNER_TEMPERATURE,
) -> Iterator[dict[str, Any]]:
    """流式决策：直接转发底层语义事件给上层 runner。

    上层 runner 据此将 ``text_delta`` 转 SSE ``token``、把 ``tool_use``
    驱动到 ``ToolRegistry.execute``。
    """
    yield from chat_completion_with_tools_stream(
        messages,
        tool_config=registry.to_bedrock_tool_config(),
        system_prompt=system_prompt,
        temperature=temperature,
    )
