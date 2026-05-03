"""International weather MCP server powered by Open-Meteo (free, no API key).

Tools exposed:
- ``get_current_weather(city)``  – current temperature, humidity, wind, feel
- ``get_weather_forecast(city, days)``  – daily high/low, rain, wind for up to 16 days
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


logger = logging.getLogger(__name__)

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 10.0

# WMO Weather interpretation codes → human-readable description
_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _wmo_desc(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return _WMO_DESCRIPTIONS.get(int(code), f"WMO code {code}")


def _geocode(city: str) -> dict[str, Any] | None:
    """返回 {name, latitude, longitude, country, timezone} 或 None。"""
    try:
        resp = httpx.get(
            _GEO_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = results[0]
        return {
            "name": r.get("name", city),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "country": r.get("country", ""),
            "timezone": r.get("timezone", "auto"),
        }
    except Exception:
        logger.exception("Geocoding failed for city: %s", city)
        return None


server = FastMCP("weather_openmeteo")


@server.tool()
def get_current_weather(city: str) -> dict[str, Any]:
    """Get the current weather conditions for any city worldwide.

    Args:
        city: City name in English, e.g. ``Auckland``, ``Shanghai``, ``London``.
    """
    loc = _geocode(city)
    if loc is None:
        return {"ok": False, "error": f"City not found: {city}"}

    try:
        resp = httpx.get(
            _FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weathercode",
                    "wind_speed_10m",
                ]),
                "timezone": loc["timezone"],
                "wind_speed_unit": "kmh",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Forecast API call failed for city: %s", city)
        return {"ok": False, "error": "Weather API request failed"}

    cur = data.get("current") or {}
    return {
        "ok": True,
        "city": loc["name"],
        "country": loc["country"],
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "temperature_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "precipitation_mm": cur.get("precipitation"),
        "wind_speed_kmh": cur.get("wind_speed_10m"),
        "condition": _wmo_desc(cur.get("weathercode")),
        "time": cur.get("time"),
    }


@server.tool()
def get_weather_forecast(city: str, days: int = 7) -> dict[str, Any]:
    """Get a multi-day weather forecast for any city worldwide.

    Args:
        city: City name in English, e.g. ``Auckland``, ``Shanghai``, ``London``.
        days: Number of forecast days (1-16). Defaults to 7.
    """
    days = max(1, min(16, days))
    loc = _geocode(city)
    if loc is None:
        return {"ok": False, "error": f"City not found: {city}"}

    try:
        resp = httpx.get(
            _FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "weathercode",
                    "wind_speed_10m_max",
                ]),
                "timezone": loc["timezone"],
                "forecast_days": days,
                "wind_speed_unit": "kmh",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Forecast API call failed for city: %s", city)
        return {"ok": False, "error": "Weather API request failed"}

    daily = data.get("daily") or {}
    times = daily.get("time") or []
    forecasts = []
    for i, date in enumerate(times):
        forecasts.append({
            "date": date,
            "temp_max_c": (daily.get("temperature_2m_max") or [])[i] if i < len(daily.get("temperature_2m_max") or []) else None,
            "temp_min_c": (daily.get("temperature_2m_min") or [])[i] if i < len(daily.get("temperature_2m_min") or []) else None,
            "precipitation_mm": (daily.get("precipitation_sum") or [])[i] if i < len(daily.get("precipitation_sum") or []) else None,
            "wind_max_kmh": (daily.get("wind_speed_10m_max") or [])[i] if i < len(daily.get("wind_speed_10m_max") or []) else None,
            "condition": _wmo_desc((daily.get("weathercode") or [])[i] if i < len(daily.get("weathercode") or []) else None),
        })

    return {
        "ok": True,
        "city": loc["name"],
        "country": loc["country"],
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "forecast_days": len(forecasts),
        "forecast": forecasts,
    }


def main() -> None:
    """以 stdio transport 启动 MCP server。"""
    server.run()


if __name__ == "__main__":  # pragma: no cover - module CLI
    main()
