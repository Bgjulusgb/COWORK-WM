"""openfootball connector — the primary free history source.

Source : https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json
License: CC0 1.0 (public domain)  ·  Key: none  ·  Account: none  ·  Rate-limit: none
        (raw.githubusercontent.com is already in the project's network allowlist)
Fallback: optional local repo clone (settings.openfootball_local_clone), then
          the deterministic openfootball_mock.

We pull the finished 2018 + 2022 tournaments and expose per-team history and
head-to-head. The JSON shape has drifted between years, so the parser accepts
both a top-level `matches` array and the `rounds[].matches` nesting, and both
`score.ft:[h,a]` and `score1`/`score2` score encodings.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import openfootball_mock
from data_sources.schemas import HistoricalMatch, MatchFixture
from data_sources.team_codes import to_code

log = structlog.get_logger("data_sources.openfootball")

_RAW_BASE = "https://raw.githubusercontent.com/openfootball/worldcup.json/master"
# 2010 and 2014 are on the same repo and double the training corpus (from ~128 to ~256 finished matches).
_HISTORY_YEARS = (2022, 2018, 2014, 2010)
# Historical results barely change — cache them for 30 days, not the default 6h.
_HISTORY_TTL_S = 30 * 24 * 3600


class OpenfootballConnector(BaseConnector):
    connector_name = "openfootball"

    async def _load_year(self, year: int) -> FetchResult:
        """Raw worldcup.json for one tournament (local clone first, then HTTP)."""
        local = settings.openfootball_local_clone.strip()
        if local:
            path = Path(local) / str(year) / "worldcup.json"
            try:
                if path.exists():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    return FetchResult(raw, "live", datetime.now(timezone.utc), self.connector_name)
            except Exception as exc:
                log.warning("openfootball_local_read_failed", path=str(path), error=str(exc))
        url = f"{_RAW_BASE}/{year}/worldcup.json"
        return await self._get_json(url, ttl_s=_HISTORY_TTL_S)

    async def _all_history(self) -> FetchResult:
        """Parsed HistoricalMatch list across all configured tournament years."""
        matches: list[HistoricalMatch] = []
        modes: list[str] = []
        latest: datetime | None = None
        for year in _HISTORY_YEARS:
            res = await self._load_year(year)
            modes.append(res.mode)
            if res.ok:
                matches.extend(_parse_worldcup_json(res.data, year))
                if res.fetched_at and (latest is None or res.fetched_at > latest):
                    latest = res.fetched_at
        if not matches:
            return FetchResult([], "error", None, self.connector_name)
        # Combined provenance: live if anything was freshly fetched, else cache.
        mode = "live" if "live" in modes else "cache"
        return FetchResult(matches, mode, latest, self.connector_name)

    async def get_historical_results(self, code: str) -> FetchResult:
        if settings.use_mock_openfootball:
            return FetchResult(openfootball_mock.historical_results(code), "mock", None, "mock")
        res = await self._all_history()
        if not res.ok:
            # Network/parse failure → keep the system running on mock data.
            return FetchResult(openfootball_mock.historical_results(code), "mock", None, "mock")
        code = code.upper()
        team_matches = [
            m for m in res.data if code in (m.home_code, m.away_code)
        ]
        team_matches.sort(key=lambda m: m.kickoff_utc, reverse=True)
        return res.replace_data(team_matches)

    async def get_fixtures(self, year: int = 2026) -> FetchResult:
        """Scheduled WM-2026 fixtures (with host city) for rest-days + travel.

        The mock has no 2026 schedule, so it returns an empty list — the
        RestTravelFactor then falls back to history-based rest and skips travel.
        """
        if settings.use_mock_openfootball:
            return FetchResult([], "mock", None, "mock")
        res = await self._load_year(year)
        if not res.ok:
            return FetchResult([], "mock", None, "mock")
        return res.replace_data(_parse_fixtures(res.data, year))

    async def get_head_to_head(self, home: str, away: str) -> FetchResult:
        if settings.use_mock_openfootball:
            return FetchResult(openfootball_mock.head_to_head(home, away), "mock", None, "mock")
        res = await self._all_history()
        if not res.ok:
            return FetchResult(openfootball_mock.head_to_head(home, away), "mock", None, "mock")
        home, away = home.upper(), away.upper()
        meetings = [
            m for m in res.data
            if {m.home_code, m.away_code} == {home, away}
        ]
        meetings.sort(key=lambda m: m.kickoff_utc, reverse=True)
        return res.replace_data(meetings)


def _iter_raw_matches(raw: Any) -> Iterable[dict]:
    if not isinstance(raw, dict):
        return
    if isinstance(raw.get("matches"), list):
        yield from raw["matches"]
    for rnd in raw.get("rounds", []) or []:
        for m in (rnd.get("matches", []) or []):
            yield m


def _team_name(team: Any) -> str | None:
    if isinstance(team, dict):
        return team.get("name") or team.get("code") or team.get("team")
    if isinstance(team, str):
        return team
    return None


def _score_pair(m: dict) -> tuple[int | None, int | None]:
    score = m.get("score")
    if isinstance(score, dict):
        ft = score.get("ft") or score.get("score")
        if isinstance(ft, list) and len(ft) >= 2:
            try:
                return int(ft[0]), int(ft[1])
            except (TypeError, ValueError):
                pass
    for hk, ak in (("score1", "score2"), ("ft1", "ft2"), ("home_score", "away_score")):
        if m.get(hk) is not None and m.get(ak) is not None:
            try:
                return int(m[hk]), int(m[ak])
            except (TypeError, ValueError):
                pass
    return None, None


def _parse_date(value: Any, year: int) -> datetime:
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(value[:10], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return datetime(year, 6, 1, tzinfo=timezone.utc)


def _venue_string(m: dict) -> str | None:
    """openfootball encodes the host inconsistently — stadium, city, or both."""
    stadium = m.get("stadium")
    if isinstance(stadium, dict):
        stadium = stadium.get("name") or stadium.get("key")
    city = m.get("city")
    parts = [p for p in (stadium, city) if isinstance(p, str) and p]
    return ", ".join(parts) or None


def _parse_fixtures(raw: Any, year: int) -> list[MatchFixture]:
    out: list[MatchFixture] = []
    for m in _iter_raw_matches(raw):
        name1, name2 = _team_name(m.get("team1")), _team_name(m.get("team2"))
        code1, code2 = to_code(name1), to_code(name2)
        if not code1 or not code2:
            continue
        h_score, a_score = _score_pair(m)
        out.append(MatchFixture(
            source="openfootball",
            tournament=f"World Cup {year}",
            competition_tier=1,
            home_code=code1, away_code=code2,
            home_name=name1 or code1, away_name=name2 or code2,
            kickoff_utc=_parse_date(m.get("date"), year),
            venue=_venue_string(m),
            home_score=h_score, away_score=a_score,
            is_finished=h_score is not None and a_score is not None,
        ))
    out.sort(key=lambda f: f.kickoff_utc)
    return out


def _parse_worldcup_json(raw: Any, year: int) -> list[HistoricalMatch]:
    out: list[HistoricalMatch] = []
    for m in _iter_raw_matches(raw):
        name1 = _team_name(m.get("team1"))
        name2 = _team_name(m.get("team2"))
        code1, code2 = to_code(name1), to_code(name2)
        if not code1 or not code2:
            continue
        h_score, a_score = _score_pair(m)
        out.append(HistoricalMatch(
            source="openfootball",
            tournament=f"World Cup {year}",
            competition_tier=1,
            home_code=code1,
            away_code=code2,
            home_name=name1 or code1,
            away_name=name2 or code2,
            kickoff_utc=_parse_date(m.get("date"), year),
            home_score=h_score,
            away_score=a_score,
            is_finished=h_score is not None and a_score is not None,
        ))
    return out


__all__ = ["OpenfootballConnector"]
