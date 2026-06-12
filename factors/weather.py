"""Weather strength factor.

Heat and humidity reduce high-intensity running and total goals — relevant for
WC 2026's summer kickoffs in the US South and Mexico. The orchestrator attaches
an Open-Meteo forecast (`WeatherInfo`) for the venue at kickoff; absent that the
factor stays neutral.

    over28        = max(0, temp_c - 28)
    humid_extra   = 0.003 if humidity > 70 else 0
    damp          = 1 - (0.010 + humid_extra) * min(over28, 12)
    home_strength = away_strength = damp           (symmetric goal suppression)
≈ -2% at 30 °C, up to ~-13% in extreme 40 °C heat. (Link & Weber 2017 — heat
lowers running performance in elite football.)
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_BASE_PER_DEG = 0.010
_HOT_FROM = 28.0
_CAP_DEG = 12.0


class WeatherFactor(Factor):
    name = "weather"
    default_weight = 0.04

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        w = ctx.weather
        temp = getattr(w, "temp_c", None) if w is not None else None
        if temp is None:
            return self._neutral(source="weather", reason="no_forecast")
        temp = float(temp)
        humidity = float(getattr(w, "humidity_pct", 0.0) or 0.0)

        over28 = max(0.0, temp - _HOT_FROM)
        if over28 <= 0:
            return self._neutral(source="weather", reason="mild_conditions")

        per_deg = _BASE_PER_DEG + (0.003 if humidity > 70 else 0.0)
        damp = max(0.85, 1.0 - per_deg * min(over28, _CAP_DEG))

        source, cached_at = source_from_provenance(ctx, "weather")
        return FactorSignal(
            name=self.name,
            home_strength=damp,
            away_strength=damp,
            weight=self.weight,
            confidence=0.50,
            available=True,
            source=source if source != "neutral" else "weather",
            cached_at=cached_at,
            kind="global",  # symmetric goal damp, not a home/away tilt
            raw_data={"temp_c": round(temp, 1), "humidity_pct": round(humidity, 0), "goal_damp": round(damp, 4)},
        )
