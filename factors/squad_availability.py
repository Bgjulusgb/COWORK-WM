"""Squad-availability strength factor.

Best-effort from Wikidata (mocked by default — see data_sources/wikidata.py).
A `SquadInfo` carries `star_players_available` (0..11-ish) and a list of
`notable_absences`. The relative strength of the two squads tilts xG:

    score_side    = star_players_available - 0.5 * len(notable_absences)
    home_strength = 1.0 + K * (score_home - score_away)      K = 0.02
    away_strength = 1.0 - K * (score_home - score_away)
A two-key-player edge (~score gap of 2) is a ±4% swing — deliberately modest,
because squad parsing is unreliable. The factor sits out (available=False)
whenever either squad's `SquadInfo.available` is false, which is the common
case for sides outside the top tier.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_K = 0.02


def _squad_score(info) -> float | None:
    if info is None or not getattr(info, "available", False):
        return None
    stars = getattr(info, "star_players_available", None)
    if stars is None:
        return None
    absences = len(getattr(info, "notable_absences", []) or [])
    return float(stars) - 0.5 * absences


class SquadAvailabilityFactor(Factor):
    name = "squad_availability"
    default_weight = 0.08

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_score = _squad_score(ctx.squad_meta_home)
        away_score = _squad_score(ctx.squad_meta_away)
        if home_score is None or away_score is None:
            return self._neutral(source="squad", reason="squad_unavailable")

        delta = home_score - away_score
        home_strength = 1.0 + _K * delta
        away_strength = 1.0 - _K * delta

        source, cached_at = source_from_provenance(ctx, "squad_home", "squad_away")
        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=0.50,  # squad data is noisy even when present
            available=True,
            source=source,
            cached_at=cached_at,
            raw_data={
                "home_score": round(home_score, 2),
                "away_score": round(away_score, 2),
                "home_stars": getattr(ctx.squad_meta_home, "star_players_available", None),
                "away_stars": getattr(ctx.squad_meta_away, "star_players_available", None),
            },
        )
