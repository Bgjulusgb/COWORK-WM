"""Deterministic climatological weather, used when Open-Meteo is mocked or the
match is beyond the live forecast horizon (~16 days).

A crude June-climate model: hotter at lower latitude, cooler at altitude. Good
enough to let the WeatherFactor fire plausibly for the hot southern venues
without any network call.
"""
from __future__ import annotations

from data_sources.schemas import WeatherInfo


def weather_for(latitude: float | None, longitude: float | None, altitude_m: float | None) -> WeatherInfo:
    lat = 30.0 if latitude is None else float(latitude)
    alt = 0.0 if altitude_m is None else float(altitude_m)
    # June daytime baseline by latitude (≈34 °C at lat 20, ≈22 °C at lat 47).
    base = 34.0 - 0.45 * (lat - 20.0)
    temp = base - 0.0065 * alt          # standard environmental lapse rate
    # Coastal/low latitudes muggier; high & inland drier.
    humidity = max(30.0, min(85.0, 70.0 - 0.5 * (lat - 25.0) - 0.01 * alt))
    return WeatherInfo(
        source="mock",
        temp_c=round(temp, 1),
        humidity_pct=round(humidity, 0),
        wind_kmh=10.0,
        precipitation_mm=0.0,
    )


__all__ = ["weather_for"]
