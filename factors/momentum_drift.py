"""Momentum / sentiment-drift strength factor.

Distinct from SentimentFactor (which uses the sentiment *level*): this reads the
*rate of change* — the per-hour trend slope from `analysis.trend_analyzer` — so
a side whose mood is rising into kickoff gets a small boost even if its absolute
sentiment is still middling. Captures "team trending up after good news".

    d_slope       = tanh(8 * (slope_home - slope_away))   ∈ [-1, 1]
    home_strength = 1.0 + 0.05 * d_slope * sample_weight
    away_strength = 1.0 - 0.05 * d_slope * sample_weight
Slopes are tiny (sentiment units / hour), hence the tanh scaling. Stays neutral
with no posts. (Builds on the existing trend/anomaly detection.)
"""
from __future__ import annotations

import math

from factors.base import Factor, FactorContext, FactorSignal

_K = 0.05
_FULL_SAMPLE = 50.0
_SLOPE_SCALE = 8.0


class MomentumDriftFactor(Factor):
    name = "momentum_drift"
    default_weight = 0.05

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        payload = ctx.sentiment_payload or {}
        sample = int(payload.get("sample_size", 0) or 0)
        if sample <= 0:
            return self._neutral(source="reddit", reason="no_posts")

        slope_home = float(payload.get("home_trend_slope", 0.0) or 0.0)
        slope_away = float(payload.get("away_trend_slope", 0.0) or 0.0)
        sample_weight = min(1.0, sample / _FULL_SAMPLE)

        d_slope = math.tanh(_SLOPE_SCALE * (slope_home - slope_away))
        home_strength = 1.0 + _K * d_slope * sample_weight
        away_strength = 1.0 - _K * d_slope * sample_weight

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=min(0.55, 0.20 + 0.5 * sample_weight),
            available=True,
            source="reddit",
            raw_data={
                "home_slope": round(slope_home, 5),
                "away_slope": round(slope_away, 5),
                "drift": round(d_slope, 3),
            },
        )
