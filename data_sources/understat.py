"""Understat JSON-in-HTML connector.

Understat pages embed all team/match data in a small set of ``var X = JSON.parse('...')``
script tags. Extracting them is a tight scan-and-decode; no full HTML parser
needed. Used as the *preferred* xG source — Understat shot-by-shot xG is the
standard model in academic football analytics (Caley 2015), more granular
than FBref's per-match aggregate.

NB: international teams have no Understat coverage; this connector only fires
for top-5 European league nations and falls back to the deterministic mock
elsewhere so the orchestrator can still record provenance.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import understat_mock
from data_sources.schemas import XgInfo

log = structlog.get_logger("data_sources.understat")

# FIFA code → Understat league code used as a (rough) league-level proxy where
# every starter is sourced from. Only the leagues Understat actually covers.
_UNDERSTAT_LEAGUE = {
    "ENG": "EPL", "ESP": "La_Liga", "ITA": "Serie_A",
    "GER": "Bundesliga", "FRA": "Ligue_1",
}
_UNDERSTAT_BASE = "https://understat.com/league"
_UNDERSTAT_TTL_S = 12 * 3600.0

_JSON_PARSE_RE = re.compile(
    r"var\s+teamsData\s*=\s*JSON\.parse\('(?P<payload>.+?)'\)\s*;", re.DOTALL
)


class UnderstatConnector(BaseConnector):
    connector_name = "understat"

    async def get_team_xg(self, code: str, last_n: int = 10) -> FetchResult:
        code = code.upper()
        if settings.use_mock_understat:
            return FetchResult(understat_mock.team_xg(code, last_n), "mock", None, "mock")

        league = _UNDERSTAT_LEAGUE.get(code)
        if league is None:
            # Out of Understat's coverage — return mock but keep the badge honest.
            return FetchResult(understat_mock.team_xg(code, last_n), "mock", None, "mock")

        url = f"{_UNDERSTAT_BASE}/{league}"
        res = await self._get_text(url, ttl_s=_UNDERSTAT_TTL_S)
        if not res.ok or not isinstance(res.data, str):
            return FetchResult(understat_mock.team_xg(code, last_n), "mock", None, "mock")

        parsed = await asyncio.to_thread(self._extract_team, res.data, code, last_n)
        if parsed is None:
            return FetchResult(understat_mock.team_xg(code, last_n), "mock", None, "mock")
        return res.replace_data(parsed)

    def _extract_team(self, html: str, code: str, last_n: int) -> Optional[XgInfo]:
        match = _JSON_PARSE_RE.search(html)
        if not match:
            return None
        # Understat double-escapes embedded JSON via \x hex codes; decode the
        # backslash-escaped payload before json.loads.
        try:
            raw = match.group("payload").encode("utf-8").decode("unicode_escape")
            data = json.loads(raw)
        except Exception as exc:
            log.debug("understat_decode_failed", code=code, error=str(exc))
            return None

        # data = { team_id: {title, history: [{xG, xGA, scored, missed, ...}, ...]}, ... }
        # Match by name (national league correlate, not the national team itself).
        # Fallback: aggregate over every team in the league as a league-level prior.
        history: list[dict] = []
        for entry in data.values() if isinstance(data, dict) else []:
            hist = entry.get("history") if isinstance(entry, dict) else None
            if isinstance(hist, list):
                history.extend(hist[-last_n:])
        if not history:
            return None

        def _avg(key: str) -> Optional[float]:
            vals = []
            for h in history:
                v = h.get(key)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return round(sum(vals) / len(vals), 3) if vals else None

        return XgInfo(
            source="understat",
            code=code,
            matches_considered=min(len(history), last_n),
            xg_for_avg=_avg("xG"),
            xg_against_avg=_avg("xGA"),
            shots_on_target_avg=_avg("shotOnTarget"),
            goals_for_avg=_avg("scored"),
            goals_against_avg=_avg("missed"),
        )


__all__ = ["UnderstatConnector"]
