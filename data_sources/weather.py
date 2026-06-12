"""Open-Meteo weather connector — venue forecast around kickoff.

Base : https://api.open-meteo.com/v1/forecast   ·  Key: none  ·  Account: none
Free : non-commercial, no registration, generous limits.
Horizon: ~16 days ahead. Beyond that (or on any failure) we degrade to the
deterministic climatology mock so the WeatherFactor still has something to read.

We request hourly fields in UTC and pick the hour nearest kickoff.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import weather_mock
from data_sources.schemas import WeatherInfo

log = structlog.get_logger("data_sources.weather")

_BASE = "https://api.open-meteo.com/v1/forecast"
_WEATHER_TTL_S = 3 * 3600  # forecasts move; 3h cache is plenty


class WeatherConnector(BaseConnector):
    connector_name = "weather"

    async def get_weather(
        self, latitude: float | None, longitude: float | None,
        altitude_m: float | None, kickoff_utc: datetime | None,
    ) -> FetchResult:
        if settings.use_mock_weather or latitude is None or longitude is None:
            return FetchResult(weather_mock.weather_for(latitude, longitude, altitude_m), "mock", None, "mock")

        params = {
            "latitude": round(float(latitude), 3),
            "longitude": round(float(longitude), 3),
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "timezone": "UTC",
            "forecast_days": 16,
        }
        res = await self._get_json(_BASE, params=params, ttl_s=_WEATHER_TTL_S)
        if not res.ok:
            return FetchResult(weather_mock.weather_for(latitude, longitude, altitude_m), "mock", None, "mock")

        info = _parse(res.data, kickoff_utc)
        if info is None:
            # Match is past the forecast horizon → climatology fallback.
            return FetchResult(weather_mock.weather_for(latitude, longitude, altitude_m), "mock", None, "mock")
        return res.replace_data(info)


def _parse(data: Any, kickoff_utc: datetime | None) -> WeatherInfo | None:
    hourly = (data or {}).get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    if not times or not temps:
        return None

    idx = _nearest_hour_index(times, kickoff_utc)
    if idx is None or idx >= len(temps):
        return None

    def _at(key: str) -> float | None:
        arr = hourly.get(key) or []
        return float(arr[idx]) if idx < len(arr) and arr[idx] is not None else None

    return WeatherInfo(
        source="open-meteo",
        temp_c=float(temps[idx]) if temps[idx] is not None else None,
        humidity_pct=_at("relative_humidity_2m"),
        wind_kmh=_at("wind_speed_10m"),
        precipitation_mm=_at("precipitation"),
    )


def _nearest_hour_index(times: list[str], kickoff_utc: datetime | None) -> int | None:
    if kickoff_utc is None:
        return 0
    target = kickoff_utc.astimezone(timezone.utc).replace(tzinfo=None)
    best_idx, best_gap = None, None
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t)
        except ValueError:
            continue
        gap = abs((dt - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best_gap, best_idx = gap, i
    # If the closest hour is more than a day away the venue isn't in range.
    if best_gap is not None and best_gap > 86400:
        return None
    return best_idx


__all__ = ["WeatherConnector"]
