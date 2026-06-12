"""Integration test for services.match_service.run_crawl_and_predict.

Uses an in-memory aiosqlite and the mock crawler so we exercise the full
crawl → score → snapshot → predict pipeline without any network calls.
"""
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings
from db.database import Base, _add_missing_columns
from db.models import (
    MatchPrediction,
    RedditPost,
    SentimentScore,
    SentimentSnapshot,
    WM2026Match,
)
from services.match_service import (
    run_crawl_and_predict,
    upsert_match_from_config,
)
from utils.config_loader import discover_match_configs


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with Session() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def first_config_path() -> Path:
    cfgs = list(discover_match_configs())
    assert cfgs, "expected at least one match config under backend/config/matches/"
    return cfgs[0]


@pytest.mark.asyncio
async def test_crawl_and_predict_produces_prediction(session, first_config_path, monkeypatch):
    # Force the mock crawler regardless of .env
    monkeypatch.setattr(settings, "use_mock_crawler", True)
    monkeypatch.setattr(settings, "use_arctic_shift", False)

    match = await upsert_match_from_config(session, first_config_path)
    await session.commit()

    posts, scored, pred = await run_crawl_and_predict(session, match, crawl_seed=42)

    assert posts > 0
    assert scored > 0
    assert isinstance(pred, MatchPrediction)
    assert 0.0 < pred.home_win_prob < 1.0
    assert 0.0 < pred.away_win_prob < 1.0
    assert 0.0 < pred.draw_prob < 1.0
    total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
    assert abs(total - 1.0) < 0.01

    snap = (await session.execute(
        SentimentSnapshot.__table__.select().where(SentimentSnapshot.match_id == match.id)
    )).first()
    assert snap is not None


@pytest.mark.asyncio
async def test_run_is_idempotent_on_duplicate_posts(session, first_config_path, monkeypatch):
    monkeypatch.setattr(settings, "use_mock_crawler", True)
    monkeypatch.setattr(settings, "use_arctic_shift", False)

    match = await upsert_match_from_config(session, first_config_path)
    await session.commit()

    await run_crawl_and_predict(session, match, crawl_seed=7)
    # Second crawl with same seed → mock returns identical post_ids → existing
    # ones should be skipped (posts_added stays low or zero).
    posts2, scored2, _ = await run_crawl_and_predict(session, match, crawl_seed=7)
    assert posts2 == 0
    assert scored2 == 0
