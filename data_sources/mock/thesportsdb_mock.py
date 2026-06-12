"""Synthetic team metadata + venue info for when TheSportsDB is mocked."""
from __future__ import annotations

from data_sources.schemas import TeamMeta, VenueInfo
from data_sources.team_codes import preferred_name
from scripts.team_real_data import get_world_ranking


def team_meta(code: str) -> TeamMeta:
    code = code.upper()
    return TeamMeta(
        source="mock",
        code=code,
        name=preferred_name(code),
        logo_url=None,
        fifa_world_ranking=get_world_ranking(code),
        elo_rating=None,
        founded_year=None,
    )


def venue(name: str | None) -> VenueInfo:
    return VenueInfo(
        source="mock",
        venue_id=None,
        name=name or "Unknown Venue",
        city=None,
        country=None,
        latitude=None,
        longitude=None,
        altitude_m=None,
        capacity=None,
    )


__all__ = ["team_meta", "venue"]
