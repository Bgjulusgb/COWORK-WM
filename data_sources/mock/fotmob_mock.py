"""Deterministic FotMob mock: lineup + structured injuries per team.

Real FotMob returns a confirmed XI ~1h before kickoff; offline we always emit
a plausible "season-average" XI with two of the eleven flagged as bench-promotions
for the third of teams whose seed hash demands it, plus one injury for a sixth
of teams. Enough variation for the LineupStrengthFactor to fire deterministically.
"""
from __future__ import annotations

import hashlib

from data_sources.mock import team_strength
from data_sources.schemas import LineupInfo, StructuredInjury


def _seed(code: str) -> int:
    return int(hashlib.md5(("fotmob:" + code.upper()).encode("utf-8")).hexdigest(), 16)


def _starters(code: str) -> list[str]:
    return [f"{code}_player_{i:02d}" for i in range(1, 12)]


def lineup(code: str) -> LineupInfo:
    code = code.upper()
    h = _seed(code)
    strength = team_strength(code)
    avg_value = 280_000_000 * strength       # WC sides ≈ €0.15–0.7B XI value
    promotions = 2 if h % 3 == 0 else (1 if h % 5 == 0 else 0)
    # If the XI has promotions, its actual value drops below the season avg.
    starter_value = avg_value * (1.0 - 0.08 * promotions)
    return LineupInfo(
        source="mock",
        code=code,
        is_confirmed=(h % 2 == 0),
        starters=_starters(code),
        starters_value_eur=round(starter_value, 2),
        season_avg_value_eur=round(avg_value, 2),
        bench_promotions=promotions,
    )


def injuries(code: str) -> list[StructuredInjury]:
    code = code.upper()
    h = _seed(code)
    out: list[StructuredInjury] = []
    if h % 6 == 0:
        out.append(StructuredInjury(
            source="mock", team_code=code,
            player=f"{code}_player_07", position="MF",
            status="doubt", severity=0.4,
        ))
    if h % 11 == 0:
        out.append(StructuredInjury(
            source="mock", team_code=code,
            player=f"{code}_player_09", position="FW",
            status="out", severity=0.85,
        ))
    return out


__all__ = ["lineup", "injuries"]
