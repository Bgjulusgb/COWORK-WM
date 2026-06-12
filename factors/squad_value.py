"""Squad-value factor (Transfermarkt).

log(home_value / away_value) is a robust market-implied talent gap (Peeters
2018; Gerhards & Mutz 2017). We use the **top-11 value** rather than the full
squad — international matches are decided by starters, not depth.

The factor *complements* the Elo/FIFA-ranking pair (which captures *results*)
with a wisdom-of-crowds **talent** estimate. They typically agree but diverge
for U-23 talents on hot streaks (high Elo, low TM value) and aging stars
(low form, high TM value), which is where this factor moves the needle.
"""
from __future__ import annotations

import math

from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_MAX_NUDGE = 0.10   # cap at ±10 % per side


def _top11(info) -> float | None:
    if info is None:
        return None
    v = getattr(info, "top11_value_eur", None) or getattr(info, "total_value_eur", None)
    if not v or v <= 0:
        return None
    return float(v)


class SquadValueFactor(Factor):
    name = "squad_value"
    default_weight = 0.0    # off until Transfermarkt connector live or weight tuned

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_v = _top11(ctx.squad_value_home)
        away_v = _top11(ctx.squad_value_away)
        if home_v is None or away_v is None:
            return self._neutral(source="transfermarkt", reason="missing_value")

        # log-ratio in [~-3, ~3] for realistic WC matchups (€100M vs €2B → log10≈1.3).
        log_ratio = math.log10(home_v / away_v)
        clipped = max(-1.0, min(1.0, log_ratio))
        home_strength = 1.0 + _MAX_NUDGE * clipped
        away_strength = 1.0 - _MAX_NUDGE * clipped

        # Confidence rises with absolute talent gap *and* with sample (both
        # squads above €100M ⇒ trustworthy figures).
        gap = abs(clipped)
        body = min(home_v, away_v)
        size_factor = min(1.0, body / 200_000_000)
        confidence = max(0.4, min(0.9, 0.5 + 0.3 * gap + 0.2 * size_factor))

        source, cached_at = source_from_provenance(ctx, "squad_value_home", "squad_value_away")
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
                "home_top11_eur": int(home_v),
                "away_top11_eur": int(away_v),
                "log10_ratio": round(log_ratio, 3),
            },
        )


__all__ = ["SquadValueFactor"]
