"""MCP 客户端工厂：加载真实外部 MCP server 的工具。

设计要点：
- 使用 ``langchain_mcp_adapters.client.MultiServerMCPClient`` 一次性
  加载多个 server（stdio）暴露的工具，并自动适配为 LangChain Tool。
- 加载流程：
  1. 从 ``backend.config`` 读 ``MCP_*_COMMAND`` / ``MCP_*_ARGS`` /
     ``MCP_*_ENABLED`` 拼出 ``MultiServerMCPClient`` 的 server 配置。
  2. 用一个新建的事件循环（独立线程）运行 ``client.get_tools()``，
     避免与 FastAPI 现有事件循环冲突。
  3. 把异步 BaseTool 包成 ``StructuredTool`` 的同步版本，便于 LangGraph
     ``ToolNode`` 在同步 ``.stream()`` 路径里调用。
- 失败容忍：任何环节出问题（``langchain-mcp-adapters`` 未安装、
  Node 不可用、stdio server 启动失败）都会被吞成 ``MCPLoadResult([], None)``，
  ``ExternalContextAgent`` 内部会把它当成"没拿到外部工具"，依然能跑完
  整条多 Agent 流水线（只是没有外部上下文）。
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from backend.config import (
    BRAVE_API_KEY,
    MCP_BRAVE_SEARCH_ARGS,
    MCP_BRAVE_SEARCH_COMMAND,
    MCP_BRAVE_SEARCH_ENABLED,
    MCP_BUSINESS_CALENDAR_ARGS,
    MCP_BUSINESS_CALENDAR_COMMAND,
    MCP_BUSINESS_CALENDAR_ENABLED,
    MCP_WEATHER_ARGS,
    MCP_WEATHER_COMMAND,
    MCP_WEATHER_ENABLED,
)


logger = logging.getLogger(__name__)


@dataclass
class MCPLoadResult:
    """加载结果。``client`` 持有 MultiServerMCPClient 引用，runtime 必须把
    本对象挂在自身生命周期内的字段上，避免 client 被 GC 后底层 stdio
    session 提前关闭。
    """

    tools: list[BaseTool] = field(default_factory=list)
    client: Any | None = None
    enabled_servers: list[str] = field(default_factory=list)
    failed_servers: list[str] = field(default_factory=list)


def _split_args(args_str: str) -> list[str]:
    """把环境变量里的字符串拆成 argv list。支持空白分隔与引号转义。"""
    text = (args_str or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()


def build_mcp_server_config() -> dict[str, dict[str, Any]]:
    """根据 env 描述符拼出 MultiServerMCPClient 的 server config。

    返回的 dict 形如 ``{name: {"command": ..., "args": [...], "transport": "stdio"}}``。
    单个 server 缺命令时直接跳过，不进入返回值。
    """
    config: dict[str, dict[str, Any]] = {}

    if MCP_WEATHER_ENABLED and MCP_WEATHER_COMMAND.strip():
        config["weather"] = {
            "command": MCP_WEATHER_COMMAND.strip(),
            "args": _split_args(MCP_WEATHER_ARGS),
            "transport": "stdio",
        }

    if MCP_BRAVE_SEARCH_ENABLED and MCP_BRAVE_SEARCH_COMMAND.strip():
        env: dict[str, str] = {}
        if BRAVE_API_KEY:
            env["BRAVE_API_KEY"] = BRAVE_API_KEY
        entry: dict[str, Any] = {
            "command": MCP_BRAVE_SEARCH_COMMAND.strip(),
            "args": _split_args(MCP_BRAVE_SEARCH_ARGS),
            "transport": "stdio",
        }
        if env:
            entry["env"] = env
        config["brave_search"] = entry

    if MCP_BUSINESS_CALENDAR_ENABLED and MCP_BUSINESS_CALENDAR_COMMAND.strip():
        config["business_calendar"] = {
            "command": MCP_BUSINESS_CALENDAR_COMMAND.strip(),
            "args": _split_args(MCP_BUSINESS_CALENDAR_ARGS),
            "transport": "stdio",
        }

    return config


def _wrap_async_tool_sync(async_tool: BaseTool) -> StructuredTool:
    """把 async-only 的 MCP BaseTool 适配成同步可调用的 StructuredTool。

    LangGraph 的 ``ToolNode`` 在 ``graph.stream(...)`` 路径下走同步分支，
    若原工具只实现了 ``_arun``，会因找不到 ``_run`` 抛 ``NotImplementedError``。
    我们用一个独立线程内的新事件循环运行协程，避免与 FastAPI 主循环冲突。
    """

    name = getattr(async_tool, "name", "mcp_tool")
    description = getattr(async_tool, "description", "") or ""
    args_schema = getattr(async_tool, "args_schema", None)

    async def _ainvoke(**kwargs: Any) -> Any:
        return await async_tool.ainvoke(kwargs)

    def _invoke(**kwargs: Any) -> Any:
        # 在独立线程里跑独立 event loop，无论调用方是否已有 loop 都安全。
        result_box: dict[str, Any] = {}
        error_box: dict[str, BaseException] = {}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result_box["value"] = loop.run_until_complete(_ainvoke(**kwargs))
            except BaseException as exc:  # noqa: BLE001
                error_box["error"] = exc
            finally:
                try:
                    loop.close()
                except Exception:  # pragma: no cover - defensive
                    pass

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("value")

    return StructuredTool.from_function(
        func=_invoke,
        coroutine=_ainvoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )


async def _load_tools_async(config: dict[str, dict[str, Any]]):
    """异步加载工具：按 server 隔离地 catch，单个失败不影响其它。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    enabled: list[str] = []
    failed: list[str] = []
    tools: list[BaseTool] = []

    # 逐个 server 尝试加载，便于在某个 server 启动失败时记录并跳过。
    for name, server_cfg in config.items():
        try:
            single_client = MultiServerMCPClient({name: server_cfg})
            server_tools = await single_client.get_tools()
            tools.extend(server_tools)
            enabled.append(name)
        except Exception:  # noqa: BLE001
            logger.exception("MCP server %s failed to load", name)
            failed.append(name)

    return tools, enabled, failed


def _run_async(coro) -> Any:
    """在独立线程内的全新 event loop 里运行协程，返回结果。

    避免与可能存在的运行中 loop（如 uvicorn 启动期）冲突。
    """
    box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            try:
                loop.close()
            except Exception:  # pragma: no cover - defensive
                pass

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def load_external_mcp_tools() -> MCPLoadResult:
    """启动期一次性加载所有外部 MCP server 暴露的工具。

    任何环节失败都会被吞成空结果，调用方据此决定降级策略。
    """
    try:
        import langchain_mcp_adapters  # noqa: F401
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters 未安装，跳过外部 MCP 工具加载。"
        )
        return MCPLoadResult()

    config = build_mcp_server_config()
    if not config:
        logger.info("没有启用任何 MCP server，外部工具集为空。")
        return MCPLoadResult()

    try:
        async_tools, enabled, failed = _run_async(_load_tools_async(config))
    except Exception:  # noqa: BLE001
        logger.exception("加载 MCP 工具时出现意外错误，外部工具集降级为空。")
        return MCPLoadResult()

    sync_tools = [_wrap_async_tool_sync(t) for t in async_tools]
    return MCPLoadResult(
        tools=sync_tools,
        client=None,
        enabled_servers=enabled,
        failed_servers=failed,
    )
