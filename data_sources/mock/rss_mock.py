"""Deterministic injury-news mock.

Real injury feeds are sparse and noisy, so offline we synthesise a stable,
small set: roughly a third of teams carry a minor doubt, a few a ruled-out
starter. Seeded by code so tests and repeated runs agree.
"""
from __future__ import annotations

import hashlib

from data_sources.schemas import InjuryNewsItem


def _seed(code: str) -> int:
    return int(hashlib.md5(code.upper().encode("utf-8")).hexdigest(), 16)


def injury_news(code: str) -> list[InjuryNewsItem]:
    code = code.upper()
    h = _seed(code)
    items: list[InjuryNewsItem] = []
    if h % 3 == 0:
        items.append(InjuryNewsItem(
            source="mock", team_code=code,
            headline=f"{code}: midfielder rated a doubt with a minor knock", impact=0.4,
        ))
    if h % 7 == 0:
        items.append(InjuryNewsItem(
            source="mock", team_code=code,
            headline=f"{code}: first-choice striker ruled out", impact=0.8,
        ))
    return items


__all__ = ["injury_news"]
