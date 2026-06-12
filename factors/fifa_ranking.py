"""FIFA-world-ranking strength factor.

Complements Elo: where Elo is continuous and result-driven, the FIFA ranking
encodes the confederation-weighted, longer-horizon view. Reads the ranking from
TeamMeta (TheSportsDB) and falls back to the bundled `team_real_data` table so
it is always available.

    delta         = away_rank - home_rank      (positive ⇒ home better; lower = better)
    home_strength = 1.0 + K * clip(delta, -60, 60)     K = 0.0015
A #1-vs-#48 gap (delta≈47) is ~+7% xG for the favourite — a deliberately small,
high-confidence nudge that mostly agrees with Elo.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_K = 0.0015
_CLIP = 60


def _rank(meta, code: str) -> int | None:
    if meta is not None and getattr(meta, "fifa_world_ranking", None):
        return int(meta.fifa_world_ranking)
    try:
        from scripts.team_real_data import get_world_ranking

        return int(get_world_ranking(code.upper()))
    except Exception:
        return None


class FifaRankingFactor(Factor):
    name = "fifa_ranking"
    default_weight = 0.05

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_rank = _rank(ctx.team_meta_home, ctx.home_code)
        away_rank = _rank(ctx.team_meta_away, ctx.away_code)
        if home_rank is None or away_rank is None:
            return self._neutral(source="fifa", reason="missing_ranking")

        delta = max(-_CLIP, min(_CLIP, away_rank - home_rank))
        home_strength = 1.0 + _K * delta
        away_strength = 1.0 - _K * delta

        source, cached_at = source_from_provenance(ctx, "team_meta_home", "team_meta_away")
        if source == "neutral":
            source = "ranking_table"
        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=0.70,
            available=True,
            source=source,
            cached_at=cached_at,
            raw_data={"home_rank": home_rank, "away_rank": away_rank, "rank_delta": delta},
        )
