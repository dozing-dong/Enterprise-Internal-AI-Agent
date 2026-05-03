"""Business Calendar MCP server.

通过 ``python -m backend.mcp_servers.business_calendar`` 启动一个 stdio MCP
server，对外暴露三个工具：

- ``is_business_day(date, country)``：判断给定日期是否为该国家工作日。
- ``add_business_days(start_date, days, country)``：从某天起跳过节假日 / 周末
  累加 N 个工作日，返回结果日期。
- ``country_holidays(year, country)``：列出某国全年节假日。

底层使用 ``holidays`` 包，默认 country = NZ（新西兰，与示例 query 一致）。
"""
