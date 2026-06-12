"""DataSourceOrchestrator — fills a FactorContext from the external connectors.

Called by match_service before the factor fan-out. Every connector runs in
parallel and is individually guarded: a failing source leaves its context field
empty (the factor then falls back to YAML or marks itself unavailable, and the
ensemble re-normalises). The orchestrator never raises into the caller.

It also records per-source provenance on ctx.provenance so each FactorSignal —
and ultimately the DataSourceBadge in the UI — can show live / cached / mock.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.fbref import FbrefConnector
from data_sources.football_data_org import FootballDataOrgConnector
from data_sources.fotmob import FotMobConnector
from data_sources.openfootball import OpenfootballConnector
from data_sources.openligadb import OpenLigaDBConnector
from data_sources.rss_news import RssNewsConnector
from data_sources.schemas import VenueInfo
from data_sources.sofascore import SofaScoreConnector
from data_sources.thesportsdb import TheSportsDBConnector
from data_sources.transfermarkt import TransfermarktConnector
from data_sources.understat import UnderstatConnector
from data_sources.venues import resolve as resolve_venue, travel_between
from data_sources.weather import WeatherConnector
from data_sources.wikidata import WikidataConnector
from factors.base import FactorContext
from utils.cache import cache

log = structlog.get_logger("data_sources.orchestrator")

# Fields populate() writes — used to snapshot/restore the per-match cache so a
# repeated crawl reuses the external data instead of re-fanning-out (D2).
_CACHED_FIELDS = (
    "historical_matches_home", "historical_matches_away", "head_to_head",
    "team_meta_home", "team_meta_away", "squad_meta_home", "squad_meta_away",
    "weather", "news_home", "news_away", "venue_info", "fixtures_for_context",
    "rest_days_home", "rest_days_away", "travel_home", "travel_away", "provenance",
    # v3.3 — new slices.
    "xg_home", "xg_away",
    "lineup_home", "lineup_away",
    "structured_injuries_home", "structured_injuries_away",
    "squad_value_home", "squad_value_away",
)


def _rest_days(matches: list[Any], kickoff: datetime | None) -> int | None:
    """Days between a team's most recent finished match and kickoff.

    Returns None when no plausible recent fixture exists (e.g. only multi-year-
    old World Cup history), so the RestTravelFactor neutralises itself rather
    than inventing a number. Clipped to [2, 10] when found.
    """
    if kickoff is None:
        return None
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    latest: datetime | None = None
    for m in matches:
        ko = getattr(m, "kickoff_utc", None)
        if ko is None or getattr(m, "home_score", None) is None:
            continue
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        if ko >= kickoff:
            continue
        if latest is None or ko > latest:
            latest = ko
    if latest is None:
        return None
    gap = (kickoff - latest).days
    if gap < 2 or gap > 60:
        return None
    return max(2, min(10, gap))


def _schedule_meta(
    fixtures: list[Any], code: str, kickoff: datetime | None, current_venue: str | None,
) -> tuple[int | None, dict[str, float] | None]:
    """From the real fixture list: (rest_days, travel) since `code`'s previous
    match before `kickoff`. Both None when there is no prior fixture (openers)."""
    if kickoff is None or not fixtures:
        return None, None
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    code = code.upper()
    prev = None
    for f in fixtures:
        if code not in (getattr(f, "home_code", ""), getattr(f, "away_code", "")):
            continue
        ko = getattr(f, "kickoff_utc", None)
        if ko is None:
            continue
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        if ko >= kickoff:
            continue
        if prev is None or ko > getattr(prev, "kickoff_utc", ko):
            prev = f
    if prev is None:
        return None, None
    prev_ko = prev.kickoff_utc
    if prev_ko.tzinfo is None:
        prev_ko = prev_ko.replace(tzinfo=timezone.utc)
    rest = max(2, min(10, (kickoff - prev_ko).days))
    travel = travel_between(getattr(prev, "venue", None), current_venue)
    return rest, travel


def _prov(res: FetchResult) -> dict[str, Any]:
    return {
        "source": res.source,
        "mode": res.mode,
        "fetched_at": res.fetched_at.isoformat() if res.fetched_at else None,
    }


def _as_result(value: Any) -> FetchResult:
    if isinstance(value, FetchResult):
        return value
    if isinstance(value, Exception):
        log.warning("orchestrator_subtask_failed", error=str(value))
    return FetchResult(None, "error")


class DataSourceOrchestrator:
    def __init__(self) -> None:
        self.openfootball = OpenfootballConnector()
        self.thesportsdb = TheSportsDBConnector()
        self.openligadb = OpenLigaDBConnector()
        self.wikidata = WikidataConnector()
        self.weather = WeatherConnector()
        self.rss = RssNewsConnector()
        self.football_data = FootballDataOrgConnector()
        # v3.3 — new connectors. All instantiated unconditionally; the use_mock_*
        # toggle inside each one switches between the mock payload and live HTTP.
        self.fbref = FbrefConnector()
        self.understat = UnderstatConnector()
        self.fotmob = FotMobConnector()
        self.sofascore = SofaScoreConnector()
        self.transfermarkt = TransfermarktConnector()

    async def populate(self, ctx: FactorContext) -> None:
        # D2: per-match cache. External data (history, squad, weather, fixtures)
        # doesn't change between two crawls of the same match, so reuse the last
        # populated snapshot within the datasource TTL instead of re-fanning-out.
        cache_key = f"orch:{ctx.match_id}"
        cached = await cache.get(cache_key)
        if cached is not None:
            for field, value in cached.items():
                setattr(ctx, field, value)
            log.debug("datasource_cache_hit", match_id=ctx.match_id)
            return

        home, away = ctx.home_code.upper(), ctx.away_code.upper()
        teams = (ctx.config or {}).get("teams") or {}
        home_name = (teams.get("home") or {}).get("name")
        away_name = (teams.get("away") or {}).get("name")

        # Resolve the venue from the static host-stadium table first — it owns
        # the authoritative coordinates/altitude the weather + altitude factors
        # need, independent of any network call.
        venue = resolve_venue(ctx.venue)
        if venue is not None:
            ctx.venue_info = VenueInfo(
                source="wc2026_venues",
                name=venue.name, city=venue.city, country=venue.country,
                latitude=venue.latitude, longitude=venue.longitude,
                altitude_m=venue.altitude_m, utc_offset_hours=venue.utc_offset_hours,
            )
        lat = venue.latitude if venue else None
        lon = venue.longitude if venue else None
        alt = venue.altitude_m if venue else None

        results = await asyncio.gather(
            self.openfootball.get_historical_results(home),
            self.openfootball.get_historical_results(away),
            self.openfootball.get_head_to_head(home, away),
            self.thesportsdb.get_team_meta(home, home_name),
            self.thesportsdb.get_team_meta(away, away_name),
            self.wikidata.get_squad_info(home),
            self.wikidata.get_squad_info(away),
            self.openligadb.get_historical_results(home),
            self.openligadb.get_historical_results(away),
            self.weather.get_weather(lat, lon, alt, ctx.kickoff_utc),
            self.rss.get_team_news(home, home_name),
            self.rss.get_team_news(away, away_name),
            self.openfootball.get_fixtures(2026),
            self.football_data.get_fixtures(2026),
            return_exceptions=True,
        )
        (hist_home, hist_away, h2h, meta_home, meta_away,
         squad_home, squad_away, olig_home, olig_away,
         weather_res, news_home, news_away, fixtures_res, fd_fixtures_res) = (_as_result(r) for r in results)

        # History — openfootball is primary; OpenLigaDB only fills a gap.
        ctx.historical_matches_home = hist_home.data or olig_home.data or []
        ctx.historical_matches_away = hist_away.data or olig_away.data or []
        ctx.head_to_head = h2h.data or []
        ctx.team_meta_home = meta_home.data
        ctx.team_meta_away = meta_away.data
        ctx.squad_meta_home = squad_home.data
        ctx.squad_meta_away = squad_away.data
        ctx.weather = weather_res.data
        ctx.news_home = news_home.data or []
        ctx.news_away = news_away.data or []

        # Rest days: prefer the real 2026 schedule (also yields travel/jet-lag);
        # fall back to the most-recent finished match in history. openfootball is
        # primary; football-data.org is the secondary/cross-check source.
        fixtures = fixtures_res.data or fd_fixtures_res.data or []
        ctx.fixtures_for_context = fixtures
        rest_home, travel_home = _schedule_meta(fixtures, home, ctx.kickoff_utc, ctx.venue)
        rest_away, travel_away = _schedule_meta(fixtures, away, ctx.kickoff_utc, ctx.venue)
        ctx.rest_days_home = rest_home if rest_home is not None else _rest_days(ctx.historical_matches_home, ctx.kickoff_utc)
        ctx.rest_days_away = rest_away if rest_away is not None else _rest_days(ctx.historical_matches_away, ctx.kickoff_utc)
        ctx.travel_home = travel_home
        ctx.travel_away = travel_away

        ctx.provenance.update({
            "history_home": _prov(hist_home if hist_home.ok else olig_home),
            "history_away": _prov(hist_away if hist_away.ok else olig_away),
            "h2h": _prov(h2h),
            "team_meta_home": _prov(meta_home),
            "team_meta_away": _prov(meta_away),
            "squad_home": _prov(squad_home),
            "squad_away": _prov(squad_away),
            "weather": _prov(weather_res),
            "news": _prov(news_home if news_home.ok else news_away),
            "fixtures": _prov(fixtures_res if fixtures_res.data else fd_fixtures_res),
        })

        # v3.3 — second fan-out for the new scrapers. Kept in its own gather so
        # a Transfermarkt 503 can't block the openfootball/weather pipeline.
        v33_results = await asyncio.gather(
            self.understat.get_team_xg(home),
            self.understat.get_team_xg(away),
            self.fbref.get_team_xg(home),
            self.fbref.get_team_xg(away),
            self.fotmob.get_lineup(home),
            self.fotmob.get_lineup(away),
            self.fotmob.get_injuries(home),
            self.fotmob.get_injuries(away),
            self.sofascore.get_lineup(home),
            self.sofascore.get_lineup(away),
            self.sofascore.get_injuries(home),
            self.sofascore.get_injuries(away),
            self.transfermarkt.get_squad_value(home),
            self.transfermarkt.get_squad_value(away),
            return_exceptions=True,
        )
        (us_home, us_away, fb_home, fb_away,
         fm_lh, fm_la, fm_ih, fm_ia,
         ss_lh, ss_la, ss_ih, ss_ia,
         tm_vh, tm_va) = (_as_result(r) for r in v33_results)

        # xG: prefer Understat (shot-level model), fall back to FBref.
        ctx.xg_home = us_home.data if us_home.ok else (fb_home.data if fb_home.ok else None)
        ctx.xg_away = us_away.data if us_away.ok else (fb_away.data if fb_away.ok else None)

        # Lineups: take whichever source has more starters.
        def _pick_lineup(a: FetchResult, b: FetchResult):
            la, lb = a.data, b.data
            if la is None:
                return lb, b
            if lb is None:
                return la, a
            return (la, a) if len(getattr(la, "starters", []) or []) >= len(getattr(lb, "starters", []) or []) else (lb, b)

        ctx.lineup_home, lineup_home_res = _pick_lineup(fm_lh, ss_lh)
        ctx.lineup_away, lineup_away_res = _pick_lineup(fm_la, ss_la)

        # Structured injuries: union FotMob + SofaScore, dedupe by (player, team).
        def _merge_injuries(a: FetchResult, b: FetchResult) -> list:
            seen: set[tuple[str, str]] = set()
            out: list = []
            for src in (a.data or [], b.data or []):
                for item in src:
                    key = (getattr(item, "team_code", ""), getattr(item, "player", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
            return out

        ctx.structured_injuries_home = _merge_injuries(fm_ih, ss_ih)
        ctx.structured_injuries_away = _merge_injuries(fm_ia, ss_ia)

        ctx.squad_value_home = tm_vh.data
        ctx.squad_value_away = tm_va.data

        ctx.provenance.update({
            "xg_home": _prov(us_home if us_home.ok else fb_home),
            "xg_away": _prov(us_away if us_away.ok else fb_away),
            "lineup_home": _prov(lineup_home_res),
            "lineup_away": _prov(lineup_away_res),
            "injuries_structured_home": _prov(fm_ih if fm_ih.ok else ss_ih),
            "injuries_structured_away": _prov(fm_ia if fm_ia.ok else ss_ia),
            "squad_value_home": _prov(tm_vh),
            "squad_value_away": _prov(tm_va),
        })

        # Snapshot the populated fields for the next crawl of this match.
        snapshot = {field: getattr(ctx, field) for field in _CACHED_FIELDS}
        await cache.set(cache_key, snapshot, settings.datasource_cache_ttl_hours * 3600.0)

        log.debug(
            "datasource_populated",
            match_id=ctx.match_id,
            hist_home=len(ctx.historical_matches_home),
            hist_away=len(ctx.historical_matches_away),
            h2h=len(ctx.head_to_head),
            history_mode=ctx.provenance["history_home"]["mode"],
            weather_mode=ctx.provenance["weather"]["mode"],
        )

    async def aclose(self) -> None:
        await BaseConnector.close_all()


__all__ = ["DataSourceOrchestrator"]
