"""``python -m backend.mcp_servers.business_calendar`` 入口。

直接调用 ``server.main()`` 启动 stdio MCP server，便于通过
``backend/config.py`` 里的 ``MCP_BUSINESS_CALENDAR_*`` 环境变量直接拉起。
"""

from backend.mcp_servers.business_calendar.server import main


if __name__ == "__main__":
    main()
