"""Business Calendar MCP server.

Run via ``python -m backend.mcp_servers.business_calendar`` to start a
stdio MCP server that exposes three tools:

- ``is_business_day(date, country)``: check whether a date is a business
  day in the given country.
- ``add_business_days(start_date, days, country)``: starting from a date,
  add N business days while skipping weekends and public holidays.
- ``country_holidays(year, country)``: list public holidays for a country
  in the given year.

Backed by the ``holidays`` package; default country = NZ (New Zealand,
matching the example query).
"""
