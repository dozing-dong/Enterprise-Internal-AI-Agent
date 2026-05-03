"""MCP（Model Context Protocol）外部工具接入层。

把 MCP 当作"外部工具执行层"暴露给多 Agent 流水线：
- ``load_external_mcp_tools()``：根据 ``backend/config.py`` 里的描述符
  通过 ``langchain-mcp-adapters`` 拉真实 MCP server（stdio）的工具列表，
  返回一份可直接 ``bind_tools`` 的 LangChain Tool 列表。
- ``MCPLoadResult``：除了 ``tools`` 还携带 ``client`` 引用，确保 MCP
  session 在 runtime 生命周期内不会被 GC，外部工具调用能持续生效。
- 任意一个 server 加载失败都会被吞成空 tool 集，永远不会阻塞主服务启动。
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
