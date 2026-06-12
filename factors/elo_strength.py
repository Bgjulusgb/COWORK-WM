"""Elo-rating strength factor.

Default weight 0.30 — the single biggest input in the ensemble. Source is the
elo_rating field that every WM-2026 YAML already carries. A future iteration
can recompute the value from openfootball historical results (K-factor update),
but the YAML is the authoritative input today.

Projection:
    delta = home_elo - away_elo, clipped to ±400 (typical international range)
    home_strength = 1.0 + delta / 1000
    away_strength = 2.0 - home_strength
A 200-Elo gap therefore yields a 0.2 swing (~+20% xG home, ~-20% away).
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class EloStrengthFactor(Factor):
    name = "elo_strength"
    default_weight = 0.30

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        teams = (ctx.config or {}).get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}

        home_elo = home.get("elo_rating")
        away_elo = away.get("elo_rating")
        if not isinstance(home_elo, (int, float)) or not isinstance(away_elo, (int, float)):
            return self._neutral(source="yaml", reason="missing_elo")

        delta = _clip(int(home_elo) - int(away_elo), -400, 400)
        home_strength = 1.0 + delta / 1000.0
        away_strength = 2.0 - home_strength

        # Confidence: high when both ratings exist; slightly lower if one of
        # them looks like a placeholder (very common when codes are TBD).
        confidence = 0.85
        if home_elo < 1300 or away_elo < 1300:
            confidence = 0.60

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=confidence,
            available=True,
            source="yaml",
            raw_data={
                "home_elo": int(home_elo),
                "away_elo": int(away_elo),
                "elo_delta": delta,
            },
        )
