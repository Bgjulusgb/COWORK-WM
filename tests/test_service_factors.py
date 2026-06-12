"""Regression guard for the v3 multi-factor path.

Before v3 the factor ensemble ran on Elo alone — sentiment was scored, stored,
and then ignored. These tests pin the fix: the full pipeline now produces a
multi-factor breakdown, and the Reddit sentiment actually moves the prediction.
All external data sources are mocked so the test is offline + deterministic.
"""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings
from db.database import Base, _add_missing_columns
from factors.base import FactorContext
from factors.registry import get_active_factors
from factors.sentiment_factor import SentimentFactor
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
    for flag in ("use_mock_openfootball", "use_mock_thesportsdb", "use_mock_openligadb",
                 "use_mock_wikidata", "use_mock_weather", "use_mock_rss"):
        monkeypatch.setattr(settings, flag, True)


def test_registry_activates_full_factor_set():
    names = {f.name for f in get_active_factors(settings)}
    # The regression was "only elo_strength". Pin the core multi-factor set.
    for required in ("elo_strength", "form", "head_to_head", "goal_efficiency", "sentiment"):
        assert required in names, f"{required} missing from active factors"


@pytest.mark.asyncio
async def test_pipeline_produces_multifactor_breakdown(session, monkeypatch):
    _force_offline(monkeypatch)
    cfgs = discover_match_configs()
    target = next((c for c in cfgs if "aus_vs_tur" in c.name), cfgs[0])
    match = await upsert_match_from_config(session, target)
    await session.commit()

    _, _, pred = await run_crawl_and_predict(session, match, crawl_seed=42)

    fb = pred.factor_breakdown
    assert fb is not None, "factor_breakdown must be persisted"
    available = [s for s in fb["signals"] if s["available"]]
    names = {s["name"] for s in available}
    assert len(available) >= 4, f"expected several active factors, got {names}"
    assert "sentiment" in names, "Reddit sentiment must reach the prediction"


@pytest.mark.asyncio
async def test_sentiment_swing_moves_prediction():
    """Same base xG + Elo, opposite sentiment ⇒ different home xG.

    This is the heart of the fixed regression: sentiment is no longer inert.
    """
    base_ctx = dict(
        match_id="m1",
        config={"teams": {"home": {"code": "HHH"}, "away": {"code": "AAA"}}},
        home_code="HHH", away_code="AAA",
        kickoff_utc=__import__("datetime").datetime(2026, 6, 20, tzinfo=__import__("datetime").timezone.utc),
    )
    pos = FactorContext(**base_ctx, sentiment_payload={
        "home_sentiment": 0.9, "away_sentiment": -0.9, "sample_size": 200})
    neg = FactorContext(**base_ctx, sentiment_payload={
        "home_sentiment": -0.9, "away_sentiment": 0.9, "sample_size": 200})

    f = SentimentFactor()
    sig_pos = await f.compute(pos)
    sig_neg = await f.compute(neg)
    assert sig_pos.home_strength > sig_neg.home_strength
