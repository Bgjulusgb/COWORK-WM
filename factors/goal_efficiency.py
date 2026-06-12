"""Goal-efficiency (xG-proxy) strength factor.

True shot-level xG isn't in the free sources, so we approximate each team's
attacking and defensive rates from historical goals and combine them the way a
bivariate-Poisson team-strength model does (Maher 1982): a team's expected goals
≈ its attack rate × the opponent's defensive leakiness.

    atk_x  = weighted avg goals scored per match
    def_x  = weighted avg goals conceded per match
    home_proxy = atk_home * def_away
    away_proxy = atk_away * def_home
    g          = sqrt(home_proxy * away_proxy)          (geometric mean)
    home_strength = home_proxy / g       away_strength = away_proxy / g
The geometric-mean normalisation makes home_strength * away_strength = 1 and
keeps both near 1.0, so the league scoring baseline cancels out. Flagged
`is_xg_proxy` so the UI can tooltip "proxy, not real xG".
"""
from __future__ import annotations

import math

from factors._history import team_rows, weighted_goal_rates
from factors.base import Factor, FactorContext, FactorSignal, source_from_provenance

_MIN_GAMES = 3
_EPS = 1e-6


class GoalEfficiencyFactor(Factor):
    name = "goal_efficiency"
    default_weight = 0.15

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        home_rows = team_rows(ctx.historical_matches_home, ctx.home_code)
        away_rows = team_rows(ctx.historical_matches_away, ctx.away_code)
        if len(home_rows) < _MIN_GAMES or len(away_rows) < _MIN_GAMES:
            return self._neutral(source="history", reason="insufficient_history")

        home_rates = weighted_goal_rates(home_rows)
        away_rates = weighted_goal_rates(away_rows)
        if home_rates is None or away_rates is None:
            return self._neutral(source="history", reason="insufficient_history")

        atk_home, def_home = home_rates
        atk_away, def_away = away_rates

        # Guard against a team with zero scored/conceded collapsing the proxy.
        home_proxy = max(_EPS, atk_home) * max(_EPS, def_away)
        away_proxy = max(_EPS, atk_away) * max(_EPS, def_home)
        g = math.sqrt(home_proxy * away_proxy)
        if g <= _EPS:
            return self._neutral(source="history", reason="degenerate_rates")

        # Clamp into a sane band *before* building the signal: a team that
        # conceded ~0 in a small window would otherwise explode the ratio past
        # the FactorSignal field bound (le=3.0) and raise instead of clamping.
        home_strength = max(0.5, min(2.0, home_proxy / g))
        away_strength = max(0.5, min(2.0, away_proxy / g))

        n = min(len(home_rows), len(away_rows))
        confidence = min(0.75, 0.25 + 0.07 * n)

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
                "home_attack": round(atk_home, 3),
                "home_defence": round(def_home, 3),
                "away_attack": round(atk_away, 3),
                "away_defence": round(def_away, 3),
                "is_xg_proxy": True,
            },
        )
