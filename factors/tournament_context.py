"""Tournament-context strength factor.

Two effects that the per-team xG averages don't capture on their own:

1. Host-nation advantage. For WC 2026 only USA / Mexico / Canada actually play
   on home soil. `scripts.team_real_data.get_home_advantage` returns a venue-
   aware boost in [0, 0.16] (Mexico at altitude gets the most). Home crowd /
   familiarity lifts the host and modestly suppresses the visitor.
   (Pollard 2008 — home advantage ≈ 0.6 goals at major-tournament venues.)
2. Knockout caginess. Elimination games are lower-scoring and tighter than
   group matches, so both sides' xG is damped a touch when the phase is a KO
   round. This shifts goal totals down without skewing the 1X2 balance.

The factor reports `available=False` for a neutral-venue group game (no host,
no KO), so the ensemble simply re-normalises it out — most WC games.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal

_KO_DAMP = 0.97


def _is_knockout(phase: str | None) -> bool:
    if not phase:
        return False
    return not phase.lower().startswith("group")


class TournamentContextFactor(Factor):
    name = "tournament_context"
    default_weight = 0.10

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        try:
            from scripts.team_real_data import get_home_advantage
        except Exception:
            return self._neutral(source="context", reason="data_unavailable")

        home_adv = float(get_home_advantage(ctx.home_code.upper(), ctx.venue) or 0.0)
        phase = ((ctx.config or {}).get("match") or {}).get("phase")
        is_ko = _is_knockout(phase)

        if home_adv <= 0.0 and not is_ko:
            return self._neutral(source="context", reason="neutral_group_venue")

        ko_damp = _KO_DAMP if is_ko else 1.0
        home_strength = (1.0 + home_adv) * ko_damp
        away_strength = (1.0 - 0.5 * home_adv) * ko_damp

        # High confidence for a concrete host boost; softer for the KO-only prior.
        confidence = 0.80 if home_adv > 0.0 else 0.55

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=confidence,
            available=True,
            source="context",
            raw_data={
                "home_advantage": round(home_adv, 3),
                "phase": phase,
                "is_knockout": is_ko,
                "venue": ctx.venue,
            },
        )
