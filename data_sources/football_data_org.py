"""football-data.org connector — fixtures/standings cross-check (opt-in).

Base : https://api.football-data.org/v4   ·  Key: free (X-Auth-Token header)
Tier : free tier ~10 req/min; the World Cup competition code is "WC".
Status: needs a key, so it is mocked by default (settings.use_mock_football_data
        / no FOOTBALL_DATA_API_KEY → returns []). It serves as a secondary
        fixture source: the orchestrator prefers openfootball and only falls back
        here, giving a cross-check / redundancy for rest-days + travel.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.schemas import MatchFixture
from data_sources.team_codes import to_code

log = structlog.get_logger("data_sources.football_data_org")

_BASE = "https://api.football-data.org/v4"
_FIXTURES_TTL_S = 6 * 3600


class FootballDataOrgConnector(BaseConnector):
    connector_name = "football_data_org"

    def _default_headers(self) -> dict[str, str]:
        headers = super()._default_headers()
        if settings.football_data_api_key:
            headers["X-Auth-Token"] = settings.football_data_api_key
        return headers

    def _disabled(self) -> bool:
        return settings.use_mock_football_data or not settings.football_data_api_key

    async def get_fixtures(self, year: int = 2026) -> FetchResult:
        if self._disabled():
            return FetchResult([], "mock", None, "mock")
        res = await self._get_json(f"{_BASE}/competitions/WC/matches", ttl_s=_FIXTURES_TTL_S)
        if not res.ok:
            return FetchResult([], "mock", None, "mock")
        return res.replace_data(_parse_matches(res.data, year))


def _team(node: Any) -> tuple[str | None, str | None]:
    """(code, name) from a football-data.org team node, preferring its TLA."""
    if not isinstance(node, dict):
        return None, None
    name = node.get("name") or node.get("shortName")
    tla = node.get("tla")
    code = to_code(name) or (tla.upper() if isinstance(tla, str) and len(tla) == 3 else None)
    return code, name


def _parse_matches(data: Any, year: int) -> list[MatchFixture]:
    out: list[MatchFixture] = []
    for m in (data or {}).get("matches", []) or []:
        hc, hn = _team(m.get("homeTeam"))
        ac, an = _team(m.get("awayTeam"))
        if not hc or not ac:
            continue
        ft = ((m.get("score") or {}).get("fullTime") or {})
        hs, as_ = ft.get("home"), ft.get("away")
        try:
            kickoff = datetime.fromisoformat((m.get("utcDate") or "").replace("Z", "+00:00"))
        except ValueError:
            kickoff = datetime(year, 6, 11, tzinfo=timezone.utc)
        out.append(MatchFixture(
            source="football_data_org",
            tournament="World Cup",
            competition_tier=1,
            home_code=hc, away_code=ac,
            home_name=hn or hc, away_name=an or ac,
            kickoff_utc=kickoff,
            venue=m.get("venue"),
            home_score=hs, away_score=as_,
            is_finished=(m.get("status") == "FINISHED"),
        ))
    out.sort(key=lambda f: f.kickoff_utc)
    return out


__all__ = ["FootballDataOrgConnector"]
