"""Network-strength (PageRank) factor.

Builds a directed graph of historical international matches:
    * nodes  = team codes
    * edges  = "A lost to B" (weighted by goal-difference × recency-decay)

PageRank over that graph yields a Massey/Colley-style strength score
(Bryan & Leise 2006). The score is **complementary** to Elo: Elo is a
sequential per-match update, PageRank is a stationary distribution over the
entire result network. Disagreements between the two are informative — a
team with a high Elo but low PageRank rank has been beating weak sides;
a team with low Elo but high PageRank has been losing to top sides.

Snapshot: ``models_ml/artifacts/network_strength.json`` (offline-built by
``scripts/refresh_network.py``). When the artifact is missing, the factor
self-disables and the ensemble re-normalises around it.
"""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings
from factors.base import Factor, FactorContext, FactorSignal

_ARTIFACT = settings.base_dir / "models_ml" / "artifacts" / "network_strength.json"
_MAX_NUDGE = 0.08


def _load_snapshot() -> dict[str, float] | None:
    """Load the PageRank score per FIFA code from the offline artifact.

    Returns None when the file is missing — typical until the user runs
    ``python -m scripts.refresh_network``. NetworkStrengthFactor then marks
    itself unavailable; the ensemble re-normalises around it.
    """
    try:
        data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k.upper(): float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


class NetworkStrengthFactor(Factor):
    name = "network_strength"
    default_weight = 0.0    # off until snapshot exists / user activates via Admin

    def __init__(self, weight: float | None = None) -> None:
        super().__init__(weight)
        # Lazy-load: read the artifact once per process. None until ready.
        self._snapshot: dict[str, float] | None = None
        self._loaded = False

    def _ensure_loaded(self) -> dict[str, float] | None:
        if not self._loaded:
            self._snapshot = _load_snapshot()
            self._loaded = True
        return self._snapshot

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        snap = self._ensure_loaded()
        if not snap:
            return self._neutral(source="network", reason="no_snapshot")

        home_score = snap.get(ctx.home_code.upper())
        away_score = snap.get(ctx.away_code.upper())
        if home_score is None or away_score is None:
            return self._neutral(source="network", reason="team_not_in_snapshot")

        # Normalise to (0,1] by rank percentile to keep nudges interpretable.
        scores = sorted(snap.values())
        total = max(1, len(scores))

        def _percentile(v: float) -> float:
            lo, hi = 0, total
            while lo < hi:
                mid = (lo + hi) // 2
                if scores[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            return lo / total

        home_pct = _percentile(home_score)
        away_pct = _percentile(away_score)
        delta = home_pct - away_pct

        home_strength = 1.0 + _MAX_NUDGE * delta
        away_strength = 1.0 - _MAX_NUDGE * delta

        ctx.network_strength_home = home_score
        ctx.network_strength_away = away_score

        confidence = max(0.4, min(0.9, 0.55 + 0.4 * abs(delta)))

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=confidence,
            available=True,
            source="network_pagerank",
            raw_data={
                "home_pagerank": round(home_score, 6),
                "away_pagerank": round(away_score, 6),
                "home_percentile": round(home_pct, 3),
                "away_percentile": round(away_pct, 3),
            },
        )


__all__ = ["NetworkStrengthFactor"]
