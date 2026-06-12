from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import RedditPost, SentimentScore, SentimentSnapshot, WM2026Match
from db.schemas import SentimentSnapshotResponse, TimelinePoint

router = APIRouter(prefix="/api/matches", tags=["sentiment"])


@router.get("/{match_id}/sentiment", response_model=SentimentSnapshotResponse)
async def latest_sentiment(match_id: str, session: AsyncSession = Depends(get_session)):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    q = (
        select(SentimentSnapshot)
        .where(SentimentSnapshot.match_id == match_id)
        .order_by(desc(SentimentSnapshot.snapshot_time))
        .limit(1)
    )
    snap = (await session.execute(q)).scalar_one_or_none()
    if snap is None:
        raise HTTPException(404, "No sentiment snapshot yet — call /crawl first")
    return snap


@router.get("/{match_id}/sentiment/timeline", response_model=List[TimelinePoint])
async def sentiment_timeline(
    match_id: str,
    hours: int = Query(72, ge=1, le=168),
    bucket_hours: int = Query(6, ge=1, le=24),
    session: AsyncSession = Depends(get_session),
):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    q = (
        select(RedditPost, SentimentScore)
        .join(SentimentScore, SentimentScore.post_id == RedditPost.id)
        .where(RedditPost.match_id == match_id)
    )
    rows = (await session.execute(q)).all()
    if not rows:
        return []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(hours=hours)
    points: list[TimelinePoint] = []

    bucket = bucket_hours
    n_buckets = max(1, hours // bucket)
    for i in range(n_buckets, 0, -1):
        bucket_end = now - timedelta(hours=(i - 1) * bucket)
        bucket_start = now - timedelta(hours=i * bucket)
        home_scores = []
        away_scores = []
        home_w = []
        away_w = []
        for p, s in rows:
            if p.created_utc is None:
                continue
            if not (bucket_start <= p.created_utc < bucket_end):
                continue
            w = s.engagement_weight or 0.0
            if p.team_attribution in ("home", "neutral"):
                home_scores.append((s.ensemble_score or 0.0) * (w or 1.0))
                home_w.append(w or 1.0)
            if p.team_attribution in ("away", "neutral"):
                away_scores.append((s.ensemble_score or 0.0) * (w or 1.0))
                away_w.append(w or 1.0)
        home_avg = sum(home_scores) / sum(home_w) if home_w else 0.0
        away_avg = sum(away_scores) / sum(away_w) if away_w else 0.0
        kickoff = match.kickoff_utc or now
        hours_to_ko = (kickoff - bucket_end).total_seconds() / 3600.0
        points.append(
            TimelinePoint(
                hours_to_kickoff=hours_to_ko,
                timestamp=bucket_end,
                home_sentiment=home_avg,
                away_sentiment=away_avg,
                home_post_count=len(home_scores),
                away_post_count=len(away_scores),
            )
        )
    return points
