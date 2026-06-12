"""M2: Bei wiederholtem Crawl darf hoechstens eine MatchPrediction-Row pro
match_id ``is_latest=True`` sein. Aeltere werden auf False demoted."""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings
from db.database import Base, _add_missing_columns
from db.models import MatchPrediction
from services.match_service import run_crawl_and_predict, upsert_match_from_config
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


def _force_offline(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_crawler", True)
    monkeypatch.setattr(settings, "use_arctic_shift", False)
    monkeypatch.setattr(settings, "use_factor_ensemble", True)
    for flag in (
        "use_mock_openfootball", "use_mock_thesportsdb", "use_mock_openligadb",
        "use_mock_wikidata", "use_mock_weather", "use_mock_rss",
        "use_mock_football_data", "use_mock_fbref", "use_mock_understat",
        "use_mock_fotmob", "use_mock_sofascore", "use_mock_transfermarkt",
    ):
        if hasattr(settings, flag):
            monkeypatch.setattr(settings, flag, True)


@pytest.mark.asyncio
async def test_match_prediction_has_is_latest_column():
    # Defensive smoke: the column exists on the ORM model.
    assert hasattr(MatchPrediction, "is_latest")


@pytest.mark.asyncio
async def test_repeated_crawl_keeps_only_one_is_latest(session, monkeypatch):
    _force_offline(monkeypatch)
    cfgs = discover_match_configs()
    target = next((c for c in cfgs if "aus_vs_tur" in c.name), cfgs[0])
    match = await upsert_match_from_config(session, target)
    await session.commit()

    await run_crawl_and_predict(session, match, crawl_seed=1)
    await run_crawl_and_predict(session, match, crawl_seed=2)
    await run_crawl_and_predict(session, match, crawl_seed=3)

    rows = (await session.execute(
        select(MatchPrediction).where(MatchPrediction.match_id == match.id)
    )).scalars().all()
    latest = [r for r in rows if r.is_latest]
    assert len(latest) == 1, f"expected exactly one is_latest row, got {len(latest)}"
    # Order: latest is the most recently generated_at row.
    rows_sorted = sorted(rows, key=lambda r: r.generated_at)
    assert rows_sorted[-1].is_latest is True
    for older in rows_sorted[:-1]:
        assert older.is_latest is False, f"old row id={older.id} still marked is_latest"
