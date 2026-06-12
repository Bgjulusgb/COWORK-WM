from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from crawler.wc2026_api import fetch_games, fetch_groups, fetch_teams
from db.models import WM2026Match

log = structlog.get_logger("services.wc_sync_service")


def _parse_datetime(val) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(val), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            continue
    return None


def _normalize_status(raw: str | None) -> str:
    if not raw:
        return "scheduled"
    r = str(raw).lower().strip()
    if r in ("finished", "completed", "ended", "ft"):
        return "finished"
    if r in ("live", "in_progress", "playing", "1h", "2h", "ht"):
        return "live"
    return "scheduled"


async def sync_wc_data(session: AsyncSession) -> dict:
    stats = {"games_synced": 0, "status_updated": 0, "scores_updated": 0, "errors": 0}

    try:
        api_games = await fetch_games()
    except Exception as exc:
        log.error("wc_sync_fetch_failed", error=str(exc))
        stats["errors"] = 1
        return stats

    if not api_games:
        log.warning("wc_sync_no_games", hint="worldcup26.ir returned empty response")
        return stats

    log.info("wc_sync_start", api_games_count=len(api_games))

    all_matches_q = select(WM2026Match)
    result = await session.execute(all_matches_q)
    db_matches = list(result.scalars().all())

    # BUG-08 fix: build a (home_code, away_code) → match index once. The old
    # nested loop scanned 104 matches for every API game = ~10k comparisons.
    by_teams: dict[tuple[str, str], WM2026Match] = {
        (m.home_team, m.away_team): m for m in db_matches
    }

    for game in api_games:
        home_code = _extract_team_code(game, "home")
        away_code = _extract_team_code(game, "away")
        if not home_code or not away_code:
            continue

        match = by_teams.get((home_code, away_code))
        if match is None:
            continue

        new_status = _normalize_status(game.get("status"))
        if new_status != match.status and new_status != "scheduled":
            match.status = new_status
            stats["status_updated"] += 1

        home_score = game.get("home_score") or game.get("homeScore")
        away_score = game.get("away_score") or game.get("awayScore")
        if home_score is not None and away_score is not None:
            try:
                h_s, a_s = int(home_score), int(away_score)
                if match.home_score != h_s or match.away_score != a_s:
                    match.home_score = h_s
                    match.away_score = a_s
                    stats["scores_updated"] += 1
            except (ValueError, TypeError):
                pass

        kickoff = _parse_datetime(game.get("datetime") or game.get("kickoff") or game.get("date"))
        if kickoff and match.kickoff_utc != kickoff:
            match.kickoff_utc = kickoff

        venue = game.get("venue") or game.get("stadium")
        if venue and isinstance(venue, str) and venue != match.venue:
            match.venue = venue

        stats["games_synced"] += 1

    await session.commit()
    log.info("wc_sync_done", **stats)
    return stats


def _extract_team_code(game: dict, side: str) -> str | None:
    team = game.get(side)
    if isinstance(team, dict):
        return (team.get("code") or team.get("fifa_code") or team.get("abbreviation") or "").upper() or None
    team_name = game.get(f"{side}_team") or game.get(f"{side}Team")
    if isinstance(team_name, str):
        return team_name.upper()[:3] or None
    return None
