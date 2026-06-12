"""Deterministic Transfermarkt squad-value mock. Seeded from strength so
plausible (€0.1B → €1.5B for top WC sides, lower for outsiders).
"""
from __future__ import annotations

import hashlib

from data_sources.mock import team_strength
from data_sources.schemas import SquadValueInfo


def _seed(code: str) -> int:
    return int(hashlib.md5(("tm:" + code.upper()).encode("utf-8")).hexdigest(), 16)


def squad_value(code: str) -> SquadValueInfo:
    code = code.upper()
    h = _seed(code)
    strength = team_strength(code)
    total = 350_000_000 + 1_100_000_000 * (strength - 0.55) / 1.55
    squad_size = 26 + (h % 5)
    avg = total / squad_size
    top11 = total * 0.65
    return SquadValueInfo(
        source="mock",
        code=code,
        total_value_eur=round(total, 2),
        squad_size=squad_size,
        avg_value_eur=round(avg, 2),
        top11_value_eur=round(top11, 2),
    )


__all__ = ["squad_value"]
