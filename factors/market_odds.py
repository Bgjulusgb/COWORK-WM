"""Bookmaker-odds strength factor.

Betting markets aggregate enormous private information and are the single best
public forecaster of football outcomes (efficient-market hypothesis; Štrumbelj
2014 on odds-implied probabilities). match_service computes vig-corrected
implied 1X2 probabilities (crawler.odds_api) and stores them on the context.

We turn the home/away supremacy into a modest λ-tilt:
    sup           = p_home - p_away                ∈ [-1, 1]
    home_strength = 1.0 + 0.15 * sup
    away_strength = 1.0 - 0.15 * sup
The coefficient is intentionally smaller than the market's full conviction
because the same implied probabilities are *also* blended into the final 1X2 as
a Bayesian prior in `predict_from_signals` — this factor only nudges the goal
expectation, the prior handles the outcome split, so the market isn't double
counted at full strength.
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal

_K = 0.15


class MarketOddsFactor(Factor):
    name = "market_odds"
    default_weight = 0.10

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        implied = ctx.market_implied
        if not implied or len(implied) != 3:
            return self._neutral(source="odds", reason="no_market")
        ph, pd, pa = (float(x) for x in implied)
        total = ph + pd + pa
        if not (0.95 <= total <= 1.05):
            return self._neutral(source="odds", reason="malformed_market")
        ph, pa = ph / total, pa / total

        sup = ph - pa
        home_strength = 1.0 + _K * sup
        away_strength = 1.0 - _K * sup

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=0.80,
            available=True,
            source="odds",
            raw_data={
                "p_home": round(ph, 3),
                "p_draw": round(pd / total, 3),
                "p_away": round(pa, 3),
                "supremacy": round(sup, 3),
            },
        )
