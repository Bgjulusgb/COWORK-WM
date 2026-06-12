"""Background jobs orchestrated by APScheduler.

Currently a single job: auto-crawl matches that are about to kick off but have
no recent sentiment snapshot. Pulls the match list from the DB, applies a
time-window filter, and triggers `run_crawl_and_predict` per match.

Designed to be idempotent — if a fresh snapshot exists within
`scheduler_min_gap_minutes`, the match is skipped.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import desc, select

from config.settings import settings
from db.database import AsyncSessionLocal
from db.models import SentimentSnapshot, WM2026Match
from services.match_service import run_crawl_and_predict

log = structlog.get_logger("services.scheduled_jobs")


async def crawl_upcoming_matches() -> dict:
    """Crawl every match that kicks off within the configured lookahead and
    hasn't been refreshed in the last min_gap_minutes."""
    lookahead_h = settings.scheduler_lookahead_hours
    min_gap = timedelta(minutes=settings.scheduler_min_gap_minutes)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_end = now + timedelta(hours=lookahead_h)

    triggered = 0
    skipped = 0
    failed = 0

    async with AsyncSessionLocal() as session:
        upcoming = (
            await session.execute(
                select(WM2026Match)
                .where(WM2026Match.kickoff_utc >= now)
                .where(WM2026Match.kickoff_utc <= window_end)
                .order_by(WM2026Match.kickoff_utc)
            )
        ).scalars().all()

        for match in upcoming:
            last = (
                await session.execute(
                    select(SentimentSnapshot)
                    .where(SentimentSnapshot.match_id == match.id)
                    .order_by(desc(SentimentSnapshot.snapshot_time))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if last and (now - last.snapshot_time) < min_gap:
                skipped += 1
                continue
            try:
                await run_crawl_and_predict(session, match)
                triggered += 1
            except Exception as exc:
                failed += 1
                log.warning("scheduled_crawl_failed", match_id=match.id, error=str(exc))

    log.info(
        "scheduled_run_complete",
        triggered=triggered,
        skipped=skipped,
        failed=failed,
        upcoming=len(upcoming),
    )
    return {"triggered": triggered, "skipped": skipped, "failed": failed}
