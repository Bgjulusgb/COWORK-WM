"""EXTEND-11: lightweight user tipping (skeleton).

Tokenless, anonymous tips keyed by a client-side UUID stored in localStorage.
Good enough to demo a leaderboard against the model. A real OAuth flow can
slot in later by replacing the `user_token` query parameter with a session
dependency — the data model stays.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import Base, get_session
from db.models import MatchPrediction, WM2026Match

router = APIRouter(prefix="/api/tips", tags=["tipping"])


class UserTip(Base):
    __tablename__ = "user_tips"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_token = Column(String(64), index=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    home_score = Column(Integer)
    away_score = Column(Integer)
    points = Column(Integer, default=0)
    submitted_at = Column(DateTime, default=func.now())


class TipIn(BaseModel):
    user_token: str = Field(..., min_length=8, max_length=64)
    home_score: int = Field(..., ge=0, le=20)
    away_score: int = Field(..., ge=0, le=20)


def _score_points(tip_h: int, tip_a: int, actual_h: int, actual_a: int) -> int:
    if tip_h == actual_h and tip_a == actual_a:
        return 3
    tip_outcome = "H" if tip_h > tip_a else "A" if tip_a > tip_h else "D"
    actual_outcome = "H" if actual_h > actual_a else "A" if actual_a > actual_h else "D"
    if tip_outcome == actual_outcome:
        return 1
    return 0


@router.post("/{match_id}")
async def submit_tip(
    match_id: str,
    payload: TipIn = Body(...),
    session: AsyncSession = Depends(get_session),
):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status == "finished":
        raise HTTPException(400, "Spiel ist bereits beendet — Tipps können nicht mehr eingereicht werden.")

    existing = (
        await session.execute(
            select(UserTip).where(UserTip.user_token == payload.user_token, UserTip.match_id == match_id)
        )
    ).scalar_one_or_none()
    if existing:
        existing.home_score = payload.home_score
        existing.away_score = payload.away_score
        existing.submitted_at = datetime.utcnow()
    else:
        session.add(
            UserTip(
                user_token=payload.user_token,
                match_id=match_id,
                home_score=payload.home_score,
                away_score=payload.away_score,
            )
        )
    await session.commit()
    return {"ok": True}


@router.get("/leaderboard")
async def leaderboard(session: AsyncSession = Depends(get_session)):
    """Compute per-user totals + compare to the model's accuracy."""
    rows = (await session.execute(select(UserTip))).scalars().all()
    by_user: dict[str, dict] = {}
    for tip in rows:
        bucket = by_user.setdefault(tip.user_token, {"user_token": tip.user_token, "tips": 0, "points": 0})
        bucket["tips"] += 1
        bucket["points"] += int(tip.points or 0)

    sorted_rows = sorted(by_user.values(), key=lambda r: (-r["points"], -r["tips"]))
    return {"users": sorted_rows[:50]}


async def score_pending_tips(session: AsyncSession, match_id: str, home_score: int, away_score: int) -> int:
    tips = (
        await session.execute(select(UserTip).where(UserTip.match_id == match_id))
    ).scalars().all()
    for t in tips:
        t.points = _score_points(t.home_score, t.away_score, home_score, away_score)
    await session.flush()
    return len(tips)
