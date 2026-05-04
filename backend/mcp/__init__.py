"""MCP (Model Context Protocol) external tool integration layer.

Exposes MCP as the "external tool execution layer" for the multi-agent
pipeline:
- ``load_external_mcp_tools()``: based on the descriptors in
  ``backend/config.py``, pulls the tool list from real MCP servers (stdio)
  via ``langchain-mcp-adapters`` and returns a LangChain Tool list ready
  for ``bind_tools``.
- ``MCPLoadResult``: in addition to ``tools``, carries a reference to the
  ``client`` so the MCP session is not GC'd during the runtime lifetime
  and external tool calls keep working.
- Any single server failure is swallowed into an empty tool set, so it
  never blocks main service startup.
"""

from backend.mcp.clients import (
    MCPLoadResult,
    build_mcp_server_config,
    load_external_mcp_tools,
)

__all__ = [
    "MCPLoadResult",
    "build_mcp_server_config",
    "load_external_mcp_tools",
]
