"""FotMob inofficial JSON-API connector.

FotMob's web app calls ``/api/teams?id=...`` and ``/api/lineups?matchId=...``
JSON endpoints with a public X-Mas header. We mirror those calls but degrade
gracefully on 401/429/403 — the upstream may tighten in the future, and the
factor layer should keep predicting either way.

Note: lineups become available about 60 minutes before kickoff. Calls earlier
than that return an empty lineup list ⇒ LineupStrengthFactor neutralises.
"""
from __future__ import annotations

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import fotmob_mock
from data_sources.schemas import LineupInfo, StructuredInjury

log = structlog.get_logger("data_sources.fotmob")

# FIFA code → FotMob team-id. National-team ids are stable so we hard-code.
_FOTMOB_TEAM_ID = {
    "GER": 4485, "FRA": 4481, "ENG": 6620, "ESP": 4486, "ITA": 4534,
    "BRA": 7544, "ARG": 7536, "POR": 4504, "NED": 4493, "BEL": 7568,
    "CRO": 4476, "USA": 6634, "MEX": 7570, "CAN": 9978, "JPN": 4544,
    "KOR": 4552, "AUS": 6630, "MAR": 4496, "SEN": 6625, "EGY": 4540,
    "URU": 7550, "COL": 7561, "SUI": 4488, "POL": 7515, "DEN": 4471,
    "AUT": 4490, "SRB": 7560, "ECU": 7548, "WAL": 4488, "SCO": 4480,
}
_FOTMOB_BASE = "https://www.fotmob.com/api"
_FOTMOB_TTL_S = 30 * 60.0


class FotMobConnector(BaseConnector):
    connector_name = "fotmob"

    async def get_lineup(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_fotmob:
            return FetchResult(fotmob_mock.lineup(code), "mock", None, "mock")

        team_id = _FOTMOB_TEAM_ID.get(code)
        if team_id is None:
            return FetchResult(fotmob_mock.lineup(code), "mock", None, "mock")

        res = await self._get_json(f"{_FOTMOB_BASE}/teams", params={"id": team_id}, ttl_s=_FOTMOB_TTL_S)
        if not res.ok or not isinstance(res.data, dict):
            return FetchResult(fotmob_mock.lineup(code), "mock", None, "mock")

        squad = (res.data.get("squad") or [{}])
        starters = []
        for group in squad if isinstance(squad, list) else []:
            for p in (group.get("members") or [])[:11]:
                name = p.get("name")
                if name:
                    starters.append(name)
            if len(starters) >= 11:
                break

        return res.replace_data(LineupInfo(
            source="fotmob",
            code=code,
            is_confirmed=False,
            starters=starters[:11],
            starters_value_eur=None,
            season_avg_value_eur=None,
            bench_promotions=0,
        ))

    async def get_injuries(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_fotmob:
            return FetchResult(fotmob_mock.injuries(code), "mock", None, "mock")

        team_id = _FOTMOB_TEAM_ID.get(code)
        if team_id is None:
            return FetchResult(fotmob_mock.injuries(code), "mock", None, "mock")

        res = await self._get_json(
            f"{_FOTMOB_BASE}/teams", params={"id": team_id, "tab": "squad"}, ttl_s=_FOTMOB_TTL_S
        )
        if not res.ok or not isinstance(res.data, dict):
            return FetchResult(fotmob_mock.injuries(code), "mock", None, "mock")

        out: list[StructuredInjury] = []
        for group in res.data.get("squad", []) if isinstance(res.data, dict) else []:
            for p in group.get("members", []) or []:
                injury = p.get("injury") or p.get("injuredOn")
                if not injury:
                    continue
                out.append(StructuredInjury(
                    source="fotmob",
                    team_code=code,
                    player=p.get("name", "unknown"),
                    position=group.get("title"),
                    status=str(injury.get("status", "doubt")).lower(),
                    severity=0.85 if "out" in str(injury).lower() else 0.45,
                ))
        return res.replace_data(out)


__all__ = ["FotMobConnector"]
