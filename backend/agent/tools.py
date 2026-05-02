"""工具协议与注册表。

设计要点：
- 每个工具实现 ``Tool`` 协议（name / description / input_schema / invoke）。
- ``ToolRegistry`` 同时承担两件事：
  1. 给 ``AgentRunner`` 提供按名查找与执行（带超时与异常封装）。
  2. 生成 Bedrock Converse ``toolConfig``，让 LLM 知道有哪些工具可用。
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from backend.agent.schemas import ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    """所有工具必须实现的协议。

    - ``name``: 全局唯一的工具名，下发给 LLM 用于路由。
    - ``description``: 自然语言描述，影响模型选择哪个工具，需要写得清楚。
    - ``input_schema``: 标准 JSON Schema，Bedrock toolConfig 直接消费。
    - ``invoke``: 实际执行逻辑，返回结构化的 ``ToolResult``。

    ``context`` 用来传递与请求绑定的状态（比如 session_id），
    具体哪些键由 runner 与 tool 双方约定，避免在每个工具里重复签名。
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def invoke(
        self,
        arguments: dict[str, Any],
        *,
        context: dict[str, Any],
        tool_use_id: str,
    ) -> ToolResult: ...


class ToolRegistry:
    """工具注册表。

    线程安全级别仅保证“注册一次后只读”，不支持并发动态注册。
    业务通常在应用启动时一次性 register，运行期内只读。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_bedrock_tool_config(self) -> dict[str, Any]:
        """生成 Bedrock Converse 期望的 toolConfig 结构。

        Bedrock 要求每个 tool 有 ``toolSpec.{name, description, inputSchema.json}``。
        """
        tool_specs: list[dict[str, Any]] = []
        for tool in self._tools.values():
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
            )
        return {"tools": tool_specs}

    def execute(
        self,
        call: ToolCall,
        *,
        context: dict[str, Any],
    ) -> ToolResult:
        """按名查找并执行工具，统一封装异常与耗时。

        - 工具不存在：返回 ``ok=False`` 的 ToolResult，让模型可以自行换路径。
        - 工具内部抛错：兜成 ``ok=False`` + ``error``，避免 agent 整体崩掉。
        """
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                tool_use_id=call.tool_use_id,
                name=call.name,
                ok=False,
                error=f"unknown tool: {call.name}",
            )

        start = time.perf_counter()
        try:
            result = tool.invoke(
                call.arguments,
                context=context,
                tool_use_id=call.tool_use_id,
            )
            if not isinstance(result, ToolResult):
                raise TypeError(
                    f"tool '{call.name}' must return ToolResult, "
                    f"got {type(result).__name__}"
                )
            if result.latency_ms is None:
                result.latency_ms = int((time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001 - 故意吞下，转成 ToolResult
            return ToolResult(
                tool_use_id=call.tool_use_id,
                name=call.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
