"""Deterministic FBref xG mock — seeded by team code so tests stay stable.

Real FBref pulls a HTML table of the last N matches; offline we synthesise an
xG aggregate that scales with the team's FIFA-ranking proxy so stronger sides
look more efficient. Numbers are plausible per-90 averages for international
football (1.2–2.4 xG/90 for top sides, 0.6–1.4 for mid-table).
"""
from __future__ import annotations

import hashlib

from data_sources.mock import team_strength
from data_sources.schemas import XgInfo


def _seed(code: str) -> int:
    return int(hashlib.md5(("fbref:" + code.upper()).encode("utf-8")).hexdigest(), 16)


def team_xg(code: str, last_n: int = 10) -> XgInfo:
    code = code.upper()
    h = _seed(code)
    strength = team_strength(code)
    # Anchor at strength=1.0 → 1.4 xG/90, scale linearly with strength.
    xg_for = round(0.7 + 0.7 * strength + (h % 13) * 0.02, 3)
    xg_against = round(2.2 - 1.0 * strength + (h % 11) * 0.02, 3)
    return XgInfo(
        source="mock",
        code=code,
        matches_considered=last_n,
        xg_for_avg=xg_for,
        xg_against_avg=max(0.2, xg_against),
        shots_on_target_avg=round(2.5 + 2.0 * strength, 2),
        goals_for_avg=round(xg_for * 0.95, 3),
        goals_against_avg=round(max(0.2, xg_against) * 0.95, 3),
    )


__all__ = ["team_xg"]
