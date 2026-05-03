"""Business Calendar MCP server 实现。

设计要点：
- 使用 ``mcp`` 官方 Python SDK 的 ``FastMCP`` 高阶 API，三个工具直接以
  Python 函数注册即可。
- ``holidays`` 包按 ``country`` code 选择对应国家假日表（默认 NZ）。
  对未知 country 自动降级为只判断周末。
- 所有日期均按 ISO 8601 字符串处理（YYYY-MM-DD），便于跨语言调用。
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP


logger = logging.getLogger(__name__)


_DEFAULT_COUNTRY = "NZ"


def _get_country_holidays(country: str, year: int | None = None):
    """按 country code 取一份 ``holidays`` 表；找不到则回退为空字典。"""
    try:
        import holidays as holidays_pkg
    except ImportError:  # pragma: no cover - hard dependency
        logger.exception("holidays package not installed")
        return {}

    code = (country or _DEFAULT_COUNTRY).upper()
    try:
        if year is not None:
            return holidays_pkg.country_holidays(code, years=[year])
        return holidays_pkg.country_holidays(code)
    except (NotImplementedError, KeyError):
        logger.warning("unknown country code for holidays: %s", code)
        return {}


def _parse_iso_date(value: str) -> _dt.date:
    return _dt.date.fromisoformat(value.strip())


server = FastMCP("business_calendar")


@server.tool()
def is_business_day(date: str, country: str = _DEFAULT_COUNTRY) -> dict[str, Any]:
    """Check whether the given ISO date is a business day in the given country.

    A business day is defined as Monday-Friday and not a public holiday.

    Args:
        date: ISO 8601 date string, e.g. ``2026-05-08``.
        country: ISO 3166-1 alpha-2 country code; defaults to ``NZ``.
    """
    parsed = _parse_iso_date(date)
    weekday = parsed.weekday()  # Monday=0
    is_weekend = weekday >= 5

    holidays_map = _get_country_holidays(country, parsed.year)
    holiday_name = holidays_map.get(parsed)

    result = not is_weekend and holiday_name is None
    if not result:
        if is_weekend:
            reason = "weekend"
        elif holiday_name:
            reason = f"public holiday: {holiday_name}"
        else:
            reason = "non-business day"
    else:
        reason = "business day"

    return {
        "date": parsed.isoformat(),
        "country": (country or _DEFAULT_COUNTRY).upper(),
        "is_business_day": result,
        "weekday": parsed.strftime("%A"),
        "reason": reason,
    }


@server.tool()
def add_business_days(
    start_date: str,
    days: int,
    country: str = _DEFAULT_COUNTRY,
) -> dict[str, Any]:
    """Add N business days to a start date, skipping weekends and holidays.

    Args:
        start_date: ISO 8601 start date.
        days: Number of business days to add (must be >= 0).
        country: ISO 3166-1 alpha-2 country code; defaults to ``NZ``.
    """
    if days < 0:
        return {
            "ok": False,
            "error": "days must be non-negative",
        }

    current = _parse_iso_date(start_date)
    holidays_map = _get_country_holidays(country)

    added = 0
    while added < days:
        current = current + _dt.timedelta(days=1)
        if current.weekday() >= 5:
            continue
        if current in holidays_map:
            continue
        added += 1

    return {
        "ok": True,
        "start_date": _parse_iso_date(start_date).isoformat(),
        "result_date": current.isoformat(),
        "country": (country or _DEFAULT_COUNTRY).upper(),
        "business_days_added": days,
    }


@server.tool()
def country_holidays(
    year: int,
    country: str = _DEFAULT_COUNTRY,
) -> dict[str, Any]:
    """List public holidays for a country in the given year.

    Args:
        year: 4-digit year, e.g. ``2026``.
        country: ISO 3166-1 alpha-2 country code; defaults to ``NZ``.
    """
    holidays_map = _get_country_holidays(country, year)
    items = [
        {"date": d.isoformat(), "name": str(name)}
        for d, name in sorted(holidays_map.items())
    ]
    return {
        "country": (country or _DEFAULT_COUNTRY).upper(),
        "year": year,
        "holidays": items,
    }


def main() -> None:
    """以 stdio transport 启动 MCP server。"""
    server.run()


if __name__ == "__main__":  # pragma: no cover - module CLI
    main()
