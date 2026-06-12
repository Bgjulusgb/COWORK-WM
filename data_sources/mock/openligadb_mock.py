"""Synthetic OpenLigaDB cross-check data.

OpenLigaDB is a low-weight secondary source (mostly German/European football),
so the mock just reuses the openfootball synthetic history for European teams
and returns nothing for everyone else.
"""
from __future__ import annotations

from data_sources.mock import openfootball_mock
from data_sources.schemas import HistoricalMatch

# Codes OpenLigaDB plausibly has coverage for.
_EUROPEAN = {"GER", "AUT", "SUI", "ENG", "ESP", "FRA", "POR", "NED", "BEL",
             "CRO", "SWE", "NOR", "SCO", "CZE", "BIH"}


def historical_results(code: str, n: int = 6) -> list[HistoricalMatch]:
    if code.upper() not in _EUROPEAN:
        return []
    matches = openfootball_mock.historical_results(code, n=n)
    for m in matches:
        m.source = "mock"
        m.tournament = "Mock OpenLigaDB"
    return matches


__all__ = ["historical_results"]
