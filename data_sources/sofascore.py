"""SofaScore inofficial JSON-API connector.

Sister source to FotMob (Abschnitt 2b in the plan). SofaScore exposes
``api.sofascore.com/api/v1/team/{id}/...`` which carries the same shape of
injury / lineup data. Used by the orchestrator as a cross-source validator —
if FotMob and SofaScore disagree on a lineup, ``LineupStrengthFactor`` lowers
its confidence.
"""
from __future__ import annotations

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import sofascore_mock
from data_sources.schemas import LineupInfo, StructuredInjury

log = structlog.get_logger("data_sources.sofascore")

# FIFA → SofaScore team id (national teams). Hand-curated subset; teams not
# listed fall back to mock so the badge stays honest.
_SOFA_TEAM_ID = {
    "GER": 4711, "FRA": 4481, "ENG": 4710, "ESP": 4712, "ITA": 4713,
    "BRA": 4819, "ARG": 4820, "POR": 4498, "NED": 4493, "BEL": 4757,
    "CRO": 4476, "USA": 4634, "MEX": 4570, "CAN": 4978, "JPN": 4544,
    "KOR": 4552, "AUS": 4630, "MAR": 4496, "SEN": 4625, "EGY": 4540,
}
_SOFA_BASE = "https://api.sofascore.com/api/v1"
_SOFA_TTL_S = 30 * 60.0


class SofaScoreConnector(BaseConnector):
    connector_name = "sofascore"

    def _default_headers(self) -> dict[str, str]:
        # SofaScore expects a Referer + a desktop UA — without them the API
        # 403s. Mirror what the web app sends.
        return {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer": "https://www.sofascore.com/",
            "Accept": "application/json",
        }

    async def get_lineup(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_sofascore:
            return FetchResult(sofascore_mock.lineup(code), "mock", None, "mock")

        team_id = _SOFA_TEAM_ID.get(code)
        if team_id is None:
            return FetchResult(sofascore_mock.lineup(code), "mock", None, "mock")

        res = await self._get_json(f"{_SOFA_BASE}/team/{team_id}/players", ttl_s=_SOFA_TTL_S)
        if not res.ok or not isinstance(res.data, dict):
            return FetchResult(sofascore_mock.lineup(code), "mock", None, "mock")

        players = res.data.get("players") or []
        starters = [
            (p.get("player") or {}).get("name")
            for p in players[:11]
            if isinstance(p, dict)
        ]
        starters = [s for s in starters if s]
        return res.replace_data(LineupInfo(
            source="sofascore",
            code=code,
            is_confirmed=False,
            starters=starters,
            starters_value_eur=None,
            season_avg_value_eur=None,
            bench_promotions=0,
        ))

    async def get_injuries(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_sofascore:
            return FetchResult(sofascore_mock.injuries(code), "mock", None, "mock")

        team_id = _SOFA_TEAM_ID.get(code)
        if team_id is None:
            return FetchResult(sofascore_mock.injuries(code), "mock", None, "mock")

        res = await self._get_json(f"{_SOFA_BASE}/team/{team_id}/injuries", ttl_s=_SOFA_TTL_S)
        if not res.ok or not isinstance(res.data, dict):
            return FetchResult(sofascore_mock.injuries(code), "mock", None, "mock")

        out: list[StructuredInjury] = []
        for entry in res.data.get("injuries", []) if isinstance(res.data, dict) else []:
            player = (entry.get("player") or {}).get("name", "unknown")
            status = str(entry.get("status", "doubt")).lower()
            severity = 0.85 if status in ("out", "ruled_out") else 0.45
            out.append(StructuredInjury(
                source="sofascore",
                team_code=code,
                player=player,
                status=status,
                severity=severity,
            ))
        return res.replace_data(out)


__all__ = ["SofaScoreConnector"]
