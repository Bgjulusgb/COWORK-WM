"""Synthetic squad availability for when Wikidata is mocked (the default).

Returns available=False so the SquadAvailabilityFactor stays neutral and the
ensemble re-normalises it out — mirroring real life, where we usually can't
get reliable squad data without scraping. Set available=True deterministically
for a few teams so tests can exercise the populated path too.
"""
from __future__ import annotations

from data_sources.schemas import SquadInfo
from scripts.team_real_data import get_world_ranking


def squad_info(code: str) -> SquadInfo:
    code = code.upper()
    rank = get_world_ranking(code)
    # Pretend we only managed to parse a useful infobox for top-20 sides.
    if rank <= 20:
        return SquadInfo(
            source="mock",
            code=code,
            squad_size=26,
            avg_age=27.0,
            star_players_available=max(0, 11 - rank // 4),
            notable_absences=[],
            available=True,
        )
    return SquadInfo(source="mock", code=code, available=False)


__all__ = ["squad_info"]
