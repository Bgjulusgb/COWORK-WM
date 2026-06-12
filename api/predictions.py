from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal, get_session
from db.models import MatchPrediction, WM2026Match
from db.schemas import CrawlTriggerResponse, PredictionResponse
from services.match_service import run_crawl_and_predict

router = APIRouter(prefix="/api/matches", tags=["predictions"])
batch_router = APIRouter(prefix="/api/crawl", tags=["predictions"])
log = structlog.get_logger("api.predictions")


# In-memory crawl status registry. Keyed by match_id. Survives only as long
# as the process — that's fine for now since /prediction polls the DB anyway.
_CRAWL_STATUS: Dict[str, Dict] = {}
# BUG-06 fix: keep a real reference to each background task. On shutdown the
# lifespan hook can await pending tasks instead of letting them dangle.
_CRAWL_TASKS: Dict[str, asyncio.Task] = {}


@router.get("/{match_id}/prediction", response_model=PredictionResponse)
async def latest_prediction(match_id: str, session: AsyncSession = Depends(get_session)):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    q = (
        select(MatchPrediction)
        .where(MatchPrediction.match_id == match_id)
        .order_by(desc(MatchPrediction.generated_at))
        .limit(1)
    )
    result = await session.execute(q)
    pred = result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(404, "No prediction yet — call /crawl first")
    return pred


async def _background_crawl(match_id: str) -> None:
    """Run the crawl in its own session. Updates the in-memory status registry
    and removes the task handle when finished (or cancelled)."""
    _CRAWL_STATUS[match_id] = {"status": "running", "match_id": match_id}
    try:
        async with AsyncSessionLocal() as session:
            match = await session.get(WM2026Match, match_id)
            if not match:
                _CRAWL_STATUS[match_id] = {"status": "error", "error": "match not found"}
                return
            posts, scored, pred = await run_crawl_and_predict(session, match)
        _CRAWL_STATUS[match_id] = {
            "status": "done",
            "match_id": match_id,
            "posts_crawled": posts,
            "posts_scored": scored,
            "prediction_id": pred.id,
        }
        log.info("crawl_done", match_id=match_id, posts=posts, scored=scored, prediction_id=pred.id)
    except asyncio.CancelledError:
        # Server shutdown mid-crawl — surface a stable status so polling clients
        # don't see "running" forever after a restart.
        _CRAWL_STATUS[match_id] = {"status": "cancelled", "match_id": match_id}
        log.warning("crawl_cancelled", match_id=match_id)
        raise
    except Exception as exc:
        _CRAWL_STATUS[match_id] = {"status": "error", "error": str(exc), "match_id": match_id}
        log.error("crawl_failed", match_id=match_id, error=str(exc))
    finally:
        _CRAWL_TASKS.pop(match_id, None)


def _spawn_background_crawl(match_id: str) -> None:
    """Create the background task and register it for graceful shutdown."""
    existing = _CRAWL_TASKS.get(match_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(_background_crawl(match_id), name=f"crawl:{match_id}")
    _CRAWL_TASKS[match_id] = task


async def shutdown_pending_crawls(timeout: float = 5.0) -> None:
    """Called from the FastAPI lifespan shutdown to drain running crawls."""
    pending = [t for t in _CRAWL_TASKS.values() if not t.done()]
    if not pending:
        return
    log.info("crawl_shutdown_waiting", pending=len(pending))
    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        for t in pending:
            if not t.done():
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        log.warning("crawl_shutdown_forced", cancelled=sum(1 for t in pending if t.cancelled()))


@router.post("/{match_id}/crawl", response_model=CrawlTriggerResponse)
async def trigger_crawl(
    match_id: str,
    background: bool = Query(True, description="Run crawl async (default) or block until done"),
    session: AsyncSession = Depends(get_session),
):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    current = _CRAWL_STATUS.get(match_id)
    if current and current.get("status") == "running":
        return CrawlTriggerResponse(
            match_id=match_id, posts_crawled=0, posts_scored=0, prediction_id=0,
        )

    log.info("crawl_started", match_id=match_id, background=background)

    if background:
        _spawn_background_crawl(match_id)
        return CrawlTriggerResponse(
            match_id=match_id, posts_crawled=0, posts_scored=0, prediction_id=0,
        )

    # Synchronous fallback
    try:
        posts, scored, pred = await run_crawl_and_predict(session, match)
    except Exception as exc:
        log.error("crawl_failed", match_id=match_id, error=str(exc))
        raise HTTPException(500, f"Crawl fehlgeschlagen: {exc}") from exc
    return CrawlTriggerResponse(
        match_id=match_id,
        posts_crawled=posts,
        posts_scored=scored,
        prediction_id=pred.id,
    )


@router.get("/{match_id}/crawl/status")
async def crawl_status(match_id: str):
    """Frontend polls this while a background crawl is running."""
    return _CRAWL_STATUS.get(match_id, {"status": "idle", "match_id": match_id})


class BatchCrawlResponse(BaseModel):
    triggered: int
    skipped: int
    lookahead_hours: int
    match_ids: list[str]


@batch_router.post("/batch", response_model=BatchCrawlResponse)
async def batch_crawl(
    lookahead_hours: int = Query(48, ge=1, le=720),
    session: AsyncSession = Depends(get_session),
):
    """EXTEND-09: schedule background crawls for all matches kicking off
    within `lookahead_hours`. Returns immediately; clients poll /crawl/status."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_end = now + timedelta(hours=lookahead_hours)
    upcoming = (
        await session.execute(
            select(WM2026Match)
            .where(WM2026Match.kickoff_utc >= now)
            .where(WM2026Match.kickoff_utc <= window_end)
            .order_by(WM2026Match.kickoff_utc)
        )
    ).scalars().all()

    triggered: list[str] = []
    skipped = 0
    for m in upcoming:
        current = _CRAWL_STATUS.get(m.id)
        if current and current.get("status") == "running":
            skipped += 1
            continue
        _spawn_background_crawl(m.id)
        triggered.append(m.id)

    log.info("batch_crawl", triggered=len(triggered), skipped=skipped, lookahead_h=lookahead_hours)
    return BatchCrawlResponse(
        triggered=len(triggered),
        skipped=skipped,
        lookahead_hours=lookahead_hours,
        match_ids=triggered,
    )
