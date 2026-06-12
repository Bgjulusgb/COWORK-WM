"""Head-to-head strength factor.

Counts the two teams' past meetings (filled by the orchestrator from
openfootball; falls back to the static `crawler.h2h_data` snapshot). H2H is a
weak, small-sample signal, so we shrink it toward neutral with Bayesian
pseudo-counts before tilting xG.

    score          = (home_wins - away_wins) / (N + 2*alpha)     alpha = 2
    home_strength  = 1.0 + C * score      (C = 0.15 → ≤ ±15% for a clean sweep)
    away_strength  = 1.0 - C * score
The pseudo-counts mean a single 1-0 historical record barely moves the line,
while a 6-0 dominance over many meetings approaches the cap.
"""
from __future__ import annotations

from typing import Any

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_ALPHA = 2.0
_C = 0.15


def _tally_from_matches(matches: list[Any], home: str, away: str) -> tuple[int, int, int, float]:
    home, away = home.upper(), away.upper()
    hw = dr = aw = 0
    goals = 0
    n = 0
    for m in matches:
        if getattr(m, "home_score", None) is None or getattr(m, "away_score", None) is None:
            continue
        mh = (getattr(m, "home_code", "") or "").upper()
        ma = (getattr(m, "away_code", "") or "").upper()
        if {mh, ma} != {home, away}:
            continue
        hs, as_ = int(m.home_score), int(m.away_score)
        goals += hs + as_
        n += 1
        # Normalise to the *factor's* home team regardless of who hosted.
        home_goals, away_goals = (hs, as_) if mh == home else (as_, hs)
        if home_goals > away_goals:
            hw += 1
        elif home_goals == away_goals:
            dr += 1
        else:
            aw += 1
    avg_goals = goals / n if n else 0.0
    return hw, dr, aw, avg_goals


class HeadToHeadFactor(Factor):
    name = "head_to_head"
    default_weight = 0.15

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home, away = ctx.home_code.upper(), ctx.away_code.upper()
        hw, dr, aw, avg_goals = _tally_from_matches(ctx.head_to_head, home, away)
        source, cached_at = source_from_provenance(ctx, "h2h")
        n = hw + dr + aw

        if n == 0:
            # Fall back to the static snapshot shipped with the repo.
            try:
                from crawler.h2h_data import lookup

                rec = lookup(home, away)
            except Exception:
                rec = None
            if not rec:
                return self._neutral(source="h2h", reason="no_h2h")
            hw, dr, aw = rec["home_wins"], rec["draws"], rec["away_wins"]
            avg_goals = rec["avg_goals"]
            source, cached_at = "snapshot", None
            n = hw + dr + aw
            if n == 0:
                return self._neutral(source="h2h", reason="no_h2h")

        score = (hw - aw) / (n + 2.0 * _ALPHA)
        home_strength = 1.0 + _C * score
        away_strength = 1.0 - _C * score

        confidence = min(0.70, 0.20 + 0.08 * n)

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
                "home_wins": hw,
                "draws": dr,
                "away_wins": aw,
                "meetings": n,
                "avg_goals": round(avg_goals, 2),
                "shrunk_score": round(score, 3),
            },
        )
