"""TheSportsDB V1 connector — supplementary team metadata + venue.

Base : https://www.thesportsdb.com/api/v1/json/123/   (123 = public test key)
Key  : public test key, no registration  ·  Cost: free (V1)  ·  Account: none
Note : crowd-sourced, so quality varies → low factor weight, metadata only.
       V2 / livescores are premium and intentionally NOT used.
Rate : undocumented; we cache aggressively (default TTL) and degrade to mock.
"""
from __future__ import annotations

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import thesportsdb_mock
from data_sources.schemas import TeamMeta, VenueInfo
from data_sources.team_codes import preferred_name, to_code

log = structlog.get_logger("data_sources.thesportsdb")

_BASE = "https://www.thesportsdb.com/api/v1/json/123"


class TheSportsDBConnector(BaseConnector):
    connector_name = "thesportsdb"

    async def get_team_meta(self, code: str, name: str | None = None) -> FetchResult:
        code = code.upper()
        query = name or preferred_name(code)
        if settings.use_mock_thesportsdb:
            return FetchResult(thesportsdb_mock.team_meta(code), "mock", None, "mock")

        res = await self._get_json(f"{_BASE}/searchteams.php", params={"t": query})
        if not res.ok:
            return FetchResult(thesportsdb_mock.team_meta(code), "mock", None, "mock")

        teams = (res.data or {}).get("teams") or []
        chosen = _pick_national_team(teams, code, query)
        if chosen is None:
            return FetchResult(thesportsdb_mock.team_meta(code), "mock", None, "mock")

        meta = TeamMeta(
            source="thesportsdb",
            code=code,
            name=chosen.get("strTeam") or query,
            logo_url=chosen.get("strBadge") or chosen.get("strTeamBadge"),
            fifa_world_ranking=None,
            elo_rating=None,
            founded_year=_safe_int(chosen.get("intFormedYear")),
        )
        return res.replace_data(meta)

    async def get_venue(self, name: str | None) -> FetchResult:
        if settings.use_mock_thesportsdb or not name:
            return FetchResult(thesportsdb_mock.venue(name), "mock", None, "mock")
        res = await self._get_json(f"{_BASE}/searchvenues.php", params={"t": name})
        if not res.ok:
            return FetchResult(thesportsdb_mock.venue(name), "mock", None, "mock")
        venues = (res.data or {}).get("venues") or []
        if not venues:
            return FetchResult(thesportsdb_mock.venue(name), "mock", None, "mock")
        v = venues[0]
        info = VenueInfo(
            source="thesportsdb",
            venue_id=str(v.get("idVenue")) if v.get("idVenue") else None,
            name=v.get("strVenue") or name,
            city=v.get("strLocation"),
            country=v.get("strCountry"),
            capacity=_safe_int(v.get("intCapacity")),
        )
        return res.replace_data(info)


def _pick_national_team(teams: list[dict], code: str, query: str) -> dict | None:
    """Prefer a soccer national team whose name resolves to the same FIFA code."""
    soccer = [t for t in teams if (t.get("strSport") or "").lower() == "soccer"]
    pool = soccer or teams
    for t in pool:
        if to_code(t.get("strTeam") or "") == code:
            return t
    return pool[0] if pool else None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["TheSportsDBConnector"]
