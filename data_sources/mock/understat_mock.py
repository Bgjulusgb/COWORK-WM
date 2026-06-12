"""Deterministic Understat xG mock. Slightly different numbers than FBref so
the orchestrator's "prefer Understat → fallback FBref → fallback YAML" path
can be tested end-to-end with both available.
"""
from __future__ import annotations

import hashlib

from data_sources.mock import team_strength
from data_sources.schemas import XgInfo


def _seed(code: str) -> int:
    return int(hashlib.md5(("understat:" + code.upper()).encode("utf-8")).hexdigest(), 16)


def team_xg(code: str, last_n: int = 10) -> XgInfo:
    code = code.upper()
    h = _seed(code)
    strength = team_strength(code)
    xg_for = round(0.65 + 0.75 * strength + (h % 9) * 0.025, 3)
    xg_against = round(2.1 - 0.95 * strength + (h % 17) * 0.018, 3)
    return XgInfo(
        source="mock",
        code=code,
        matches_considered=last_n,
        xg_for_avg=xg_for,
        xg_against_avg=max(0.25, xg_against),
        shots_on_target_avg=round(2.3 + 2.1 * strength, 2),
        goals_for_avg=round(xg_for * 1.02, 3),
        goals_against_avg=round(max(0.25, xg_against) * 0.98, 3),
    )


__all__ = ["team_xg"]
