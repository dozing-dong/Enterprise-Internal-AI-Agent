"""Agent 决策层。

把现有 RAG 流水线封装为一个工具（rag_answer），并提供基于 Bedrock Converse
原生 toolConfig 的多步决策循环，让模型自主选择工具完成回答。

向上层暴露的核心对象：
- ``ToolRegistry``：注册并管理可用工具。
- ``AgentRunner``：非流式 / 流式两种执行入口。
- ``AgentRunResult`` 等：返回值的数据结构。
"""

from backend.agent.runner import AgentRunner
from backend.agent.schemas import (
    AgentEvent,
    AgentRunResult,
    AgentStep,
    ToolCall,
    ToolResult,
)
from backend.agent.tools import Tool, ToolRegistry

__all__ = [
    "AgentEvent",
    "AgentRunResult",
    "AgentRunner",
    "AgentStep",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
]
