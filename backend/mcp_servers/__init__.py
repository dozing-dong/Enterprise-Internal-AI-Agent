"""Project-local MCP server implementations.

Each server is an executable Python module that uses the MCP Python SDK
to expose tools to the ``langchain-mcp-adapters`` client over stdio.

Submodules:
- ``business_calendar``: business-day / public-holiday tools backed by the
  ``holidays`` package; default region NZ.
"""
