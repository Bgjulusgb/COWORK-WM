"""Shared helpers for factors that chew on historical match lists.

The data-source connectors return `HistoricalMatch` objects newest-first. Form
and goal-efficiency both need the same recency- and competition-weighted view,
so the aggregation lives here once.

Weighting rationale:
    recency  : geometric decay 0.9^index — a result 10 games ago counts ~35%
               of the most recent one. (Hvattum & Arntzen 2010 show recency-
               weighted ratings beat flat averages for football.)
    tier     : a World Cup result is worth more than a friendly. We weight
               WM/EM finals 1.0, qualifiers / Nations League 0.8, friendlies 0.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_RECENCY_BASE = 0.9
_TIER_WEIGHT = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.4}
_POINTS = {"W": 3.0, "D": 1.0, "L": 0.0}


@dataclass
class _Row:
    scored: int
    conceded: int
    result: str          # "W" | "D" | "L"
    weight: float        # recency * tier


def _is_finished(m: Any) -> bool:
    return getattr(m, "home_score", None) is not None and getattr(m, "away_score", None) is not None


def _tier_weight(m: Any) -> float:
    return _TIER_WEIGHT.get(getattr(m, "competition_tier", 4), 0.5)


def team_rows(matches: list[Any], code: str, limit: int = 10) -> list[_Row]:
    """Per-match view from `code`'s perspective, newest first, recency+tier weighted."""
    code = code.upper()
    rows: list[_Row] = []
    idx = 0
    for m in matches:
        if not _is_finished(m):
            continue
        home = (getattr(m, "home_code", "") or "").upper()
        away = (getattr(m, "away_code", "") or "").upper()
        if code == home:
            gf, ga = int(m.home_score), int(m.away_score)
        elif code == away:
            gf, ga = int(m.away_score), int(m.home_score)
        else:
            continue
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        weight = (_RECENCY_BASE ** idx) * _tier_weight(m)
        rows.append(_Row(scored=gf, conceded=ga, result=result, weight=weight))
        idx += 1
        if idx >= limit:
            break
    return rows


def weighted_points_rate(rows: list[_Row]) -> float | None:
    """Fraction of the maximum achievable points (0..1), weighted. None if empty."""
    if not rows:
        return None
    total_w = sum(r.weight for r in rows)
    if total_w <= 0:
        return None
    earned = sum(_POINTS[r.result] * r.weight for r in rows)
    return earned / (3.0 * total_w)


def weighted_goal_rates(rows: list[_Row]) -> tuple[float, float] | None:
    """(attack, defence) = weighted avg goals scored / conceded per match. None if empty."""
    if not rows:
        return None
    total_w = sum(r.weight for r in rows)
    if total_w <= 0:
        return None
    attack = sum(r.scored * r.weight for r in rows) / total_w
    defence = sum(r.conceded * r.weight for r in rows) / total_w
    return attack, defence


__all__ = ["team_rows", "weighted_points_rate", "weighted_goal_rates"]
