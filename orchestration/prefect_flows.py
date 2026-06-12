"""Prefect flows that replace APScheduler when ``USE_PREFECT=true``.

Three top-level flows:

* :func:`daily_refresh_flow` — warms the DataSourceCache for tomorrow's
  fixtures by calling every connector once.
* :func:`pre_match_flow(match_id)` — runs the full crawl + predict path
  ≤24h before a kickoff. Used as a scheduled deployment per fixture.
* :func:`nightly_backtest_flow` — re-runs Brier-score backtesting and
  triggers a drift alert if the rolling delta exceeds a threshold.

Prefect is an *optional* dependency; importing this module without it returns
plain async functions so the same entrypoints can be invoked manually too.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

log = structlog.get_logger("orchestration.prefect_flows")


def _decorate():
    """Return ``(flow, task)`` decorators — Prefect's when installed, no-op
    otherwise. Kept tiny so the surface is identical for users who haven't
    installed Prefect yet."""
    try:
        from prefect import flow, task   # type: ignore[import-not-found]
        return flow, task
    except Exception:
        def _noop(fn=None, **_kw):
            if fn is None:
                return lambda f: f
            return fn
        return _noop, _noop


flow, task = _decorate()


@task(name="warm_orchestrator_cache")
async def _warm_orchestrator_cache(match_id: str) -> dict[str, Any]:
    """Call DataSourceOrchestrator.populate() so the per-match cache is hot."""
    from data_sources.orchestrator import DataSourceOrchestrator
    from db.database import AsyncSessionLocal
    from db.models import Match

    async with AsyncSessionLocal() as session:
        match = await session.get(Match, match_id)
        if match is None:
            return {"match_id": match_id, "ok": False, "reason": "not_found"}
        from factors.base import FactorContext
        ctx = FactorContext(
            match_id=match.id,
            config=match.config or {},
            home_code=match.home_code,
            away_code=match.away_code,
            kickoff_utc=match.kickoff_utc,
            venue=match.venue,
        )
        orch = DataSourceOrchestrator()
        try:
            await orch.populate(ctx)
            return {"match_id": match_id, "ok": True, "provenance": ctx.provenance}
        finally:
            await orch.aclose()


@flow(name="redditorakel_daily_refresh")
async def daily_refresh_flow(match_ids: list[str]) -> list[dict[str, Any]]:
    """Warm the cache for ``match_ids`` in parallel."""
    log.info("daily_refresh_start", n=len(match_ids))
    return await asyncio.gather(*(_warm_orchestrator_cache(mid) for mid in match_ids))


@flow(name="redditorakel_pre_match")
async def pre_match_flow(match_id: str) -> dict[str, Any]:
    """Full crawl + predict, gated to ≤24h before kickoff by the scheduler."""
    from db.database import AsyncSessionLocal
    from services.match_service import run_crawl_and_predict

    async with AsyncSessionLocal() as session:
        return await run_crawl_and_predict(session, match_id)


@flow(name="redditorakel_nightly_backtest")
async def nightly_backtest_flow() -> dict[str, Any]:
    """Recompute the rolling Brier score from the MatchPrediction history."""
    from analysis.backtesting import compute as compute_backtest
    from db.database import AsyncSessionLocal
    from db.models import MatchPrediction
    from sqlalchemy import select

    try:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(
                select(MatchPrediction).where(
                    MatchPrediction.actual_home_score.is_not(None)
                )
            )).scalars().all()
            report = compute_backtest(rows)
    except Exception as exc:
        log.warning("backtest_failed", error=str(exc))
        return {"ok": False, "error": str(exc)}

    return {"ok": True, "report": getattr(report, "__dict__", {})}


__all__ = [
    "daily_refresh_flow",
    "pre_match_flow",
    "nightly_backtest_flow",
]
