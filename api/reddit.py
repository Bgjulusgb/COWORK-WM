from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import RedditPost, SentimentScore, WM2026Match
from db.schemas import RedditPostResponse

router = APIRouter(prefix="/api/matches", tags=["reddit"])


@router.get("/{match_id}/reddit", response_model=List[RedditPostResponse])
async def reddit_posts(
    match_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0, description="Skip the first N rows — pair with limit for infinite scroll"),
    subreddit: Optional[str] = None,
    team: Optional[str] = Query(None, description="home | away | neutral"),
    min_score: Optional[float] = Query(None, description="Drop posts whose ensemble_score is below this"),
    max_score: Optional[float] = Query(None, description="Drop posts whose ensemble_score is above this"),
    session: AsyncSession = Depends(get_session),
):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    q = (
        select(RedditPost, SentimentScore)
        .join(SentimentScore, SentimentScore.post_id == RedditPost.id, isouter=True)
        .where(RedditPost.match_id == match_id)
    )
    if subreddit:
        q = q.where(RedditPost.subreddit == subreddit)
    if team:
        q = q.where(RedditPost.team_attribution == team)
    if min_score is not None:
        q = q.where(SentimentScore.ensemble_score >= min_score)
    if max_score is not None:
        q = q.where(SentimentScore.ensemble_score <= max_score)
    q = q.order_by(desc(RedditPost.created_utc)).offset(offset).limit(limit)
    rows = (await session.execute(q)).all()
    return [
        RedditPostResponse(
            id=p.id,
            subreddit=p.subreddit,
            tier=p.tier,
            title=p.title,
            body=p.body,
            score=p.score,
            upvote_ratio=p.upvote_ratio,
            num_comments=p.num_comments,
            created_utc=p.created_utc,
            author=p.author,
            team_attribution=p.team_attribution,
            ensemble_score=(s.ensemble_score if s else None),
            source=p.source,
        )
        for p, s in rows
    ]
