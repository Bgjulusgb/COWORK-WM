"""Verifies that init_db's auto-migration picks up the new MULTIFACTOR
columns/tables and is idempotent — follows the pytest_asyncio pattern from
test_service.py.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.database import Base, _add_missing_columns


@pytest_asyncio.fixture
async def fresh_engine():
    """Spin up an in-memory aiosqlite for the duration of one test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    yield engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_all_provisions_new_tables(fresh_engine):
    """factor_snapshots + data_source_cache come from the model registry,
    not from ALTER TABLE — they must appear after create_all alone."""
    from db import models  # noqa: F401  — load classes into Base.metadata

    async with fresh_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))

    assert "factor_snapshots" in tables
    assert "data_source_cache" in tables
    assert "match_predictions" in tables


@pytest.mark.asyncio
async def test_match_predictions_has_factor_breakdown_column(fresh_engine):
    from db import models  # noqa: F401

    async with fresh_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        cols = await conn.run_sync(
            lambda c: {col["name"] for col in inspect(c).get_columns("match_predictions")}
        )

    assert "factor_breakdown" in cols


@pytest.mark.asyncio
async def test_add_missing_columns_is_idempotent(fresh_engine):
    """_add_missing_columns must be a no-op when the schema is already current
    — that's the contract the production init_db() relies on."""
    from db import models  # noqa: F401

    async with fresh_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # First pass: no missing columns expected (create_all just ran).
        await conn.run_sync(_add_missing_columns)
        # Second pass: still must not raise.
        await conn.run_sync(_add_missing_columns)


@pytest.mark.asyncio
async def test_factor_snapshot_insert_roundtrip(fresh_engine):
    """A minimal write+read roundtrip on FactorSnapshot — confirms the column
    types we declared survive aiosqlite's type-affinity rules."""
    from db import models
    from db.models import FactorSnapshot, MatchPrediction, WM2026Match

    async with fresh_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(fresh_engine, expire_on_commit=False)
    async with Session() as s:
        s.add(
            WM2026Match(
                id="test_match",
                group="A",
                phase="group_stage",
                home_team="GER",
                away_team="CIV",
                home_name="Germany",
                away_name="Ivory Coast",
                config_path="-",
            )
        )
        await s.flush()
        s.add(
            FactorSnapshot(
                match_id="test_match",
                factor_name="elo_strength",
                home_strength=1.15,
                away_strength=0.87,
                weight=0.30,
                effective_weight=0.30,
                confidence=0.85,
                available=True,
                source="yaml",
                raw_data={"elo_delta": 250},
            )
        )
        await s.commit()

    async with Session() as s:
        rows = (await s.execute(select(FactorSnapshot))).scalars().all()
        assert len(rows) == 1
        assert rows[0].factor_name == "elo_strength"
        assert rows[0].available is True
        assert rows[0].raw_data == {"elo_delta": 250}
