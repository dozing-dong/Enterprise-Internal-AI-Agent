"""``python -m backend.mcp_servers.business_calendar`` entry point.

Calls ``server.main()`` directly to start the stdio MCP server, so it can
be launched from the ``MCP_BUSINESS_CALENDAR_*`` env vars in
``backend/config.py``.
"""

from backend.mcp_servers.business_calendar.server import main


if __name__ == "__main__":
    main()
