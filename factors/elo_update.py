"""World-Football-Elo rating update (Karlis/Ntzoufras-compatible K-update).

Pure functions so they can be unit-tested and reused by an offline script that
refreshes the YAML `elo_rating` fields from a chronological results feed. Not in
the live prediction path — the bundled ratings (team_real_data, updated weekly)
stay authoritative; recomputing from a sparse free history would add noise.

Formulae (eloratings.net method):
    E_home = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400))
    R'     = R + K · G · (W - E)
where W ∈ {1, 0.5, 0} (win/draw/loss), G is the margin-of-victory multiplier,
and K scales with match importance (WC final 60 > qualifier 40 > friendly 20).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Competition-tier → K weight (tier codes match data_sources.schemas).
_K_BY_TIER = {1: 60.0, 2: 40.0, 3: 30.0, 4: 20.0}


def expected_score(rating: float, opponent: float, home_field_adv: float = 0.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating + home_field_adv - opponent) / 400.0))


def _mov_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier — dampens runaway ratings from blowouts."""
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def update_match(
    home_rating: float, away_rating: float,
    home_goals: int, away_goals: int,
    *, tier: int = 4, home_field_adv: float = 0.0,
) -> tuple[float, float]:
    """Return the (home, away) ratings after one finished match."""
    k = _K_BY_TIER.get(tier, 20.0)
    e_home = expected_score(home_rating, away_rating, home_field_adv)
    if home_goals > away_goals:
        w_home = 1.0
    elif home_goals < away_goals:
        w_home = 0.0
    else:
        w_home = 0.5
    g = _mov_multiplier(home_goals - away_goals)
    delta = k * g * (w_home - e_home)
    # Zero-sum: what one side gains the other loses.
    return home_rating + delta, away_rating - delta


@dataclass
class _Result:
    home_code: str
    away_code: str
    home_goals: int
    away_goals: int
    tier: int


def recompute_from_history(
    initial: dict[str, float], matches: list[Any], *, default: float = 1500.0,
) -> dict[str, float]:
    """Fold a chronological (oldest-first) match list into updated ratings.

    `matches` items expose home_code/away_code/home_score/away_score and
    optionally competition_tier (data_sources.schemas.HistoricalMatch shape).
    """
    ratings = dict(initial)
    for m in matches:
        hs = getattr(m, "home_score", None)
        as_ = getattr(m, "away_score", None)
        if hs is None or as_ is None:
            continue
        hc = (getattr(m, "home_code", "") or "").upper()
        ac = (getattr(m, "away_code", "") or "").upper()
        if not hc or not ac:
            continue
        rh = ratings.get(hc, default)
        ra = ratings.get(ac, default)
        nh, na = update_match(rh, ra, int(hs), int(as_),
                              tier=getattr(m, "competition_tier", 4))
        ratings[hc], ratings[ac] = nh, na
    return ratings


__all__ = ["expected_score", "update_match", "recompute_from_history"]
