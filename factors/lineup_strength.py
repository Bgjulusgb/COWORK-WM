"""Lineup-strength factor.

Compares the confirmed/probable XI's market value to that side's season-average
XI. When a team has to start two reservists (bench_promotions ≥ 2) the tilt
swings toward the opponent.

Theory: starter market value is the standard market-implied proxy for a
player's expected contribution (see Peeters 2018, *International Journal of
Forecasting*, "Testing the wisdom of crowds in the field: Transfermarkt
valuations and international soccer results"). A 10 % shortfall in starting-XI
value vs. season-average maps to roughly a 5 % expected-goal drop.

Falls back to neutral when neither FotMob nor SofaScore returned a confirmed
lineup (typical earlier than 60 min pre-kickoff).
"""
from __future__ import annotations

import math

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_MAX_NUDGE = 0.08         # ±8 % per side
_BENCH_PENALTY = 0.03     # per promoted starter


def _ratio(li) -> float | None:
    """Starter / season-average value ratio. None if either is missing/0."""
    if li is None:
        return None
    starters = getattr(li, "starters_value_eur", None)
    season = getattr(li, "season_avg_value_eur", None)
    if not starters or not season or season <= 0:
        return None
    return float(starters) / float(season)


class LineupStrengthFactor(Factor):
    name = "lineup_strength"
    default_weight = 0.0    # off until live data is wired

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_ratio = _ratio(ctx.lineup_home)
        away_ratio = _ratio(ctx.lineup_away)
        if home_ratio is None and away_ratio is None:
            return self._neutral(source="fotmob", reason="no_lineup")

        # Tilt = log-ratio relative to 1.0 (season-average baseline). Clip to
        # a respectable band so a single missing star doesn't dominate.
        home_tilt = math.log(home_ratio) if home_ratio else 0.0
        away_tilt = math.log(away_ratio) if away_ratio else 0.0
        home_strength = 1.0 + _MAX_NUDGE * max(-0.5, min(0.5, home_tilt))
        away_strength = 1.0 + _MAX_NUDGE * max(-0.5, min(0.5, away_tilt))

        home_promotions = int(getattr(ctx.lineup_home, "bench_promotions", 0) or 0)
        away_promotions = int(getattr(ctx.lineup_away, "bench_promotions", 0) or 0)
        home_strength -= _BENCH_PENALTY * home_promotions
        away_strength -= _BENCH_PENALTY * away_promotions

        # Confirmation boost: a confirmed XI is more reliable than a projected one.
        home_conf = 0.6 if getattr(ctx.lineup_home, "is_confirmed", False) else 0.4
        away_conf = 0.6 if getattr(ctx.lineup_away, "is_confirmed", False) else 0.4
        confidence = (home_conf + away_conf) / 2

        source, cached_at = source_from_provenance(ctx, "lineup_home", "lineup_away")
        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=confidence,
            available=True,
            source=source,
            cached_at=cached_at,
            raw_data={
                "home_ratio": round(home_ratio or 1.0, 4),
                "away_ratio": round(away_ratio or 1.0, 4),
                "home_bench_promotions": home_promotions,
                "away_bench_promotions": away_promotions,
                "home_confirmed": getattr(ctx.lineup_home, "is_confirmed", False),
                "away_confirmed": getattr(ctx.lineup_away, "is_confirmed", False),
            },
        )


__all__ = ["LineupStrengthFactor"]
