"""项目自写的 MCP server 集合。

每个 server 都是一个可执行 Python 模块，使用 mcp Python SDK 暴露
工具给 ``langchain-mcp-adapters`` 客户端通过 stdio 调用。

子模块：
- ``business_calendar``：基于 ``holidays`` 包的工作日 / 节假日工具集，
  默认地区 NZ。
"""
