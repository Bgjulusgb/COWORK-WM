"""Reddit-sentiment strength factor.

This is the factor that re-injects the whole Reddit pipeline into the
prediction. Before it existed the factor-ensemble path ran on Elo alone, so the
crawled/scored sentiment was stored but never moved the line (the regression
fixed in v3).

Projection mirrors the validated legacy `MatchPredictor.predict()` nudge:
    eff_side  = sentiment + 0.5 * momentum          (clipped to [-1, 1])
    weight    = min(1, sample_size / 50)            (tiny samples are noise)
    strength  = 1.0 + 0.10 * eff_side * weight      (±10% at full conviction)
A separate per-side strength keeps an asymmetric mood (home euphoric, away
panicking) intact instead of collapsing it to a single delta.

Confidence falls with small samples and high controversy (fans split → the mean
sentiment is less informative). Source: Trotter (2024), UT Austin
arXiv:2412.10298 — Reddit VADER/engagement features predict football outcomes.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal

_MAX_NUDGE = 0.10
_FULL_SAMPLE = 300.0


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SentimentFactor(Factor):
    name = "sentiment"
    default_weight = 0.10

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        payload = ctx.sentiment_payload or {}
        sample = int(payload.get("sample_size", 0) or 0)
        if sample <= 0:
            return self._neutral(source="reddit", reason="no_posts")

        sample_weight = min(1.0, sample / _FULL_SAMPLE)

        eff_home = _clip(
            float(payload.get("home_sentiment", 0.0)) + 0.5 * float(payload.get("home_momentum", 0.0)),
            -1.0, 1.0,
        )
        eff_away = _clip(
            float(payload.get("away_sentiment", 0.0)) + 0.5 * float(payload.get("away_momentum", 0.0)),
            -1.0, 1.0,
        )

        home_strength = 1.0 + _MAX_NUDGE * eff_home * sample_weight
        away_strength = 1.0 + _MAX_NUDGE * eff_away * sample_weight

        controversy = max(
            float(payload.get("home_controversy", 0.0)),
            float(payload.get("away_controversy", 0.0)),
        )
        confidence = _clip(0.30 + 0.45 * sample_weight - 0.25 * controversy, 0.0, 1.0)

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=confidence,
            available=True,
            source="reddit",
            raw_data={
                "home_sentiment": round(eff_home, 4),
                "away_sentiment": round(eff_away, 4),
                "sample_size": sample,
                "sample_weight": round(sample_weight, 3),
                "controversy": round(controversy, 4),
            },
        )
