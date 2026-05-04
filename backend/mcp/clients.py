"""MCP client factory: load tools from real external MCP servers.

Design notes:
- Use ``langchain_mcp_adapters.client.MultiServerMCPClient`` to load tools
  exposed by multiple servers (stdio) at once and adapt them to LangChain Tools.
- Loading flow:
  1. Read ``MCP_*_COMMAND`` / ``MCP_*_ARGS`` / ``MCP_*_ENABLED`` from
     ``backend.config`` to assemble the ``MultiServerMCPClient`` server config.
  2. Run ``client.get_tools()`` in a fresh event loop on a separate thread
     to avoid conflicting with FastAPI's existing event loop.
  3. Wrap the async-only BaseTool into a synchronous StructuredTool so
     LangGraph's ``ToolNode`` can call it on the synchronous ``.stream()``
     path.
- Failure tolerance: any failure (``langchain-mcp-adapters`` not
  installed, Node unavailable, stdio server failed to start) is
  swallowed into ``MCPLoadResult([], None)``;
  ``ExternalContextAgent`` treats this as "no external tools obtained"
  and still completes the multi-agent pipeline (just without external context).
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import ToolException

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
    """Load result. ``client`` holds the MultiServerMCPClient reference; the
    runtime must attach this object to a field on its own lifetime so the
    client is not GC'd, which would prematurely close the underlying stdio
    session.
    """

    tools: list[BaseTool] = field(default_factory=list)
    client: Any | None = None
    enabled_servers: list[str] = field(default_factory=list)
    failed_servers: list[str] = field(default_factory=list)


def _split_args(args_str: str) -> list[str]:
    """Split an env-var string into an argv list. Supports whitespace splits and quoted escapes."""
    text = (args_str or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()


def build_mcp_server_config() -> dict[str, dict[str, Any]]:
    """Assemble the MultiServerMCPClient server config from env descriptors.

    The returned dict has the shape
    ``{name: {"command": ..., "args": [...], "transport": "stdio"}}``.
    Any single server that is missing a command is skipped.
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
    """Adapt an async-only MCP BaseTool into a synchronously-callable StructuredTool.

    LangGraph's ``ToolNode`` runs the synchronous branch under
    ``graph.stream(...)``; if the original tool only implements ``_arun``,
    it raises ``NotImplementedError`` because ``_run`` is not found.
    We run the coroutine in a fresh event loop on a separate thread to
    avoid conflicting with the FastAPI main loop.
    """

    name = getattr(async_tool, "name", "mcp_tool")
    description = getattr(async_tool, "description", "") or ""
    args_schema = getattr(async_tool, "args_schema", None)

    async def _ainvoke(**kwargs: Any) -> Any:
        try:
            return await async_tool.ainvoke(kwargs)
        except ToolException as exc:
            # ToolException (e.g. rate limit, MCP server error) is a "soft"
            # tool error. Return it as a structured result so ToolNode wraps
            # it in a ToolMessage the agent can read, instead of crashing the
            # entire subgraph.
            logger.warning("MCP tool %s returned ToolException: %s", name, exc)
            return {"ok": False, "error": str(exc)}

    def _invoke(**kwargs: Any) -> Any:
        # Run a fresh event loop in a separate thread; safe regardless of
        # whether the caller already has a loop.
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
    """Async tool loading: catch failures per server so a single failure does not affect others."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    enabled: list[str] = []
    failed: list[str] = []
    tools: list[BaseTool] = []

    # Try loading each server one by one to record and skip when any single
    # server fails to start.
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
    """Run a coroutine inside a fresh event loop on a separate thread and return its result.

    Avoids conflicting with any potentially running loop (e.g. during uvicorn startup).
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
    """Load tools exposed by all external MCP servers in one shot at startup.

    Any failure is swallowed into an empty result; callers decide on a
    degradation strategy based on that.
    """
    try:
        import langchain_mcp_adapters  # noqa: F401
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters is not installed; skipping external MCP tool loading."
        )
        return MCPLoadResult()

    config = build_mcp_server_config()
    if not config:
        logger.info("No MCP servers are enabled; the external tool set is empty.")
        return MCPLoadResult()

    try:
        async_tools, enabled, failed = _run_async(_load_tools_async(config))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while loading MCP tools; external tool set degraded to empty.")
        return MCPLoadResult()

    sync_tools = [_wrap_async_tool_sync(t) for t in async_tools]
    return MCPLoadResult(
        tools=sync_tools,
        client=None,
        enabled_servers=enabled,
        failed_servers=failed,
    )
