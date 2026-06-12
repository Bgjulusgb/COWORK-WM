"""Recent-form strength factor.

Reads each team's recent finished matches (filled by the DataSourceOrchestrator
from openfootball / OpenLigaDB, or the deterministic mock). Form is a recency-
and competition-weighted points rate in [0, 1]; the home/away gap tilts xG.

    rate          = weighted_points / (3 * weighted_games)   ∈ [0, 1]
    delta         = rate_home - rate_away                     ∈ [-1, 1]
    home_strength = 1.0 + K * delta        (K = 0.30 → up to ±30% at the extreme)
    away_strength = 1.0 - K * delta
A 5-0-0 vs 0-0-5 split (delta≈1) therefore gives the in-form side ~+30% xG.
(Hvattum & Arntzen 2010 — recency-weighted ratings beat the bookmaker baseline.)
"""
from __future__ import annotations

from factors._history import team_rows, weighted_points_rate
from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_K = 0.30
_MIN_GAMES = 2


class FormFactor(Factor):
    name = "form"
    default_weight = 0.20

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_rows = team_rows(ctx.historical_matches_home, ctx.home_code)
        away_rows = team_rows(ctx.historical_matches_away, ctx.away_code)

        if len(home_rows) < _MIN_GAMES or len(away_rows) < _MIN_GAMES:
            return self._neutral(source="history", reason="insufficient_history")

        rate_home = weighted_points_rate(home_rows)
        rate_away = weighted_points_rate(away_rows)
        if rate_home is None or rate_away is None:
            return self._neutral(source="history", reason="insufficient_history")

        delta = rate_home - rate_away
        home_strength = 1.0 + _K * delta
        away_strength = 1.0 - _K * delta

        # Confidence grows with the smaller sample; capped at 0.8 because form is
        # a noisy, mean-reverting signal even with a full window.
        n = min(len(home_rows), len(away_rows))
        confidence = min(0.80, 0.30 + 0.07 * n)

        source, cached_at = source_from_provenance(ctx, "history_home", "history_away")
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
                "home_form_rate": round(rate_home, 3),
                "away_form_rate": round(rate_away, 3),
                "home_games": len(home_rows),
                "away_games": len(away_rows),
            },
        )
