"""Deterministic SofaScore mock — cross-source validator for FotMob. Returns
slightly nudged numbers so the orchestrator can show "live" vs. "cache" via the
combination of both connectors.
"""
from __future__ import annotations

import hashlib

from data_sources.mock import team_strength
from data_sources.schemas import LineupInfo, StructuredInjury


def _seed(code: str) -> int:
    return int(hashlib.md5(("sofascore:" + code.upper()).encode("utf-8")).hexdigest(), 16)


def lineup(code: str) -> LineupInfo:
    code = code.upper()
    h = _seed(code)
    strength = team_strength(code)
    avg_value = 270_000_000 * strength
    promotions = 2 if h % 4 == 0 else (1 if h % 6 == 0 else 0)
    return LineupInfo(
        source="mock",
        code=code,
        is_confirmed=(h % 3 == 0),
        starters=[f"{code}_player_{i:02d}" for i in range(1, 12)],
        starters_value_eur=round(avg_value * (1.0 - 0.07 * promotions), 2),
        season_avg_value_eur=round(avg_value, 2),
        bench_promotions=promotions,
    )


def injuries(code: str) -> list[StructuredInjury]:
    code = code.upper()
    h = _seed(code)
    out: list[StructuredInjury] = []
    if h % 5 == 0:
        out.append(StructuredInjury(
            source="mock", team_code=code,
            player=f"{code}_player_10", position="DF",
            status="suspended", severity=0.6,
        ))
    return out


__all__ = ["lineup", "injuries"]
