from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_session
from db.models import WM2026Match
from db.schemas import MatchBase

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.get("", response_model=List[MatchBase])
async def list_matches(
    group: Optional[str] = None,
    phase: Optional[str] = Query(
        None,
        description="group_stage | round_of_32 | round_of_16 | quarter_finals | semi_finals | third_place | final",
    ),
    status: Optional[str] = Query(None, description="scheduled | live | finished"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """IMPROVE-14: pagination + status-aware sort.

    Default order: live first, then scheduled (chronological), then finished
    (most recently kicked-off first). Mirrors how a viewer typically scans
    the list — what's happening *now*, what's *next*, what's *over*.
    """
    q = select(WM2026Match)
    if group:
        q = q.where(WM2026Match.group == group.upper())
    if phase:
        q = q.where(WM2026Match.phase == phase)
    if status:
        q = q.where(WM2026Match.status == status)

    status_rank = case(
        (WM2026Match.status == "live", 0),
        (WM2026Match.status == "scheduled", 1),
        (WM2026Match.status == "finished", 2),
        else_=3,
    )
    # SQLite/Postgres-portable: rank by status, then chronological kickoff for
    # live/scheduled and reverse-chronological for finished.
    finished_sort = case(
        (WM2026Match.status == "finished", WM2026Match.kickoff_utc),
        else_=None,
    )
    upcoming_sort = case(
        (WM2026Match.status != "finished", WM2026Match.kickoff_utc),
        else_=None,
    )
    q = q.order_by(status_rank, upcoming_sort.asc(), finished_sort.desc()).offset(offset).limit(limit)
    result = await session.execute(q)
    return result.scalars().all()


@router.get("/count")
async def count_matches(
    group: Optional[str] = None,
    phase: Optional[str] = None,
    status: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """Companion to /api/matches for paginated UIs."""
    q = select(func.count(WM2026Match.id))
    if group:
        q = q.where(WM2026Match.group == group.upper())
    if phase:
        q = q.where(WM2026Match.phase == phase)
    if status:
        q = q.where(WM2026Match.status == status)
    total = (await session.execute(q)).scalar_one()
    return {"total": int(total)}


@router.get("/{match_id}", response_model=MatchBase)
async def get_match(match_id: str, session: AsyncSession = Depends(get_session)):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match
