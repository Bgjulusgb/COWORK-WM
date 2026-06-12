"""EXTEND-08: live updates via WebSocket + Server-Sent Events.

Two transports because deployment environments differ:
  - `ws://.../ws/match/{id}`  — full duplex, ideal for the React dashboard.
  - `GET /api/matches/{id}/stream` — SSE fallback that works through every
    proxy that allows long-lived HTTP/1.1 connections.

The producer side is a process-wide pub/sub keyed by match_id. The match
service publishes a `MatchUpdate` whenever a new prediction or snapshot lands;
clients receive serialized JSON frames.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from db.database import get_session
from db.models import MatchPrediction, WM2026Match

router = APIRouter(prefix="/api/matches", tags=["live"])
log = structlog.get_logger("api.live")


class _Hub:
    """Per-match fan-out for live updates. In-memory only."""

    def __init__(self) -> None:
        self._listeners: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, match_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        async with self._lock:
            self._listeners.setdefault(match_id, set()).add(q)
        return q

    async def unsubscribe(self, match_id: str, q: asyncio.Queue) -> None:
        async with self._lock:
            listeners = self._listeners.get(match_id)
            if listeners is not None:
                listeners.discard(q)
                if not listeners:
                    self._listeners.pop(match_id, None)

    async def publish(self, match_id: str, payload: dict) -> None:
        async with self._lock:
            listeners = list(self._listeners.get(match_id, ()))
        for q in listeners:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow consumer — drop the message rather than blocking
                # publishers. Live updates are by definition stale-tolerant.
                log.debug("live_queue_full", match_id=match_id)


hub = _Hub()


@router.websocket("/{match_id}/ws")
async def live_ws(websocket: WebSocket, match_id: str):
    await websocket.accept()
    q = await hub.subscribe(match_id)
    try:
        while True:
            payload = await q.get()
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("live_ws_error", match_id=match_id, error=str(exc))
    finally:
        await hub.unsubscribe(match_id, q)


async def _sse_iter(match_id: str) -> AsyncIterator[bytes]:
    q = await hub.subscribe(match_id)
    try:
        # Initial keepalive so proxies don't kill the connection early.
        yield b": ok\n\n"
        while True:
            payload = await q.get()
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
    finally:
        await hub.unsubscribe(match_id, q)


@router.get("/{match_id}/stream")
async def live_stream(match_id: str, session: AsyncSession = Depends(get_session)):
    match = await session.get(WM2026Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return StreamingResponse(_sse_iter(match_id), media_type="text/event-stream")


async def publish_prediction(session: AsyncSession, match_id: str) -> None:
    """Called by services.match_service after a prediction is committed."""
    pred = (
        await session.execute(
            select(MatchPrediction)
            .where(MatchPrediction.match_id == match_id)
            .order_by(desc(MatchPrediction.generated_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    if pred is None:
        return
    await hub.publish(
        match_id,
        {
            "type": "prediction",
            "match_id": match_id,
            "generated_at": pred.generated_at.isoformat() if pred.generated_at else None,
            "home_win_prob": pred.home_win_prob,
            "draw_prob": pred.draw_prob,
            "away_win_prob": pred.away_win_prob,
            "confidence": pred.confidence,
        },
    )
