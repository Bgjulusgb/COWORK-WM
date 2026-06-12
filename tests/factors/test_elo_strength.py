"""Tests for the EloStrengthFactor."""
from datetime import datetime, timezone

import pytest

from factors.base import FactorContext
from factors.elo_strength import EloStrengthFactor


def _ctx(home_elo, away_elo):
    return FactorContext(
        match_id="m1",
        config={
            "teams": {
                "home": {"name": "Home", "code": "HHH", "elo_rating": home_elo},
                "away": {"name": "Away", "code": "AAA", "elo_rating": away_elo},
            }
        },
        home_code="HHH",
        away_code="AAA",
        kickoff_utc=datetime(2026, 6, 20, 19, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_equal_elo_yields_neutral_signal():
    sig = await EloStrengthFactor().compute(_ctx(1850, 1850))
    assert sig.available is True
    assert sig.home_strength == pytest.approx(1.0)
    assert sig.away_strength == pytest.approx(1.0)
    assert sig.confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_home_favoured_when_elo_higher():
    sig = await EloStrengthFactor().compute(_ctx(2020, 1820))
    # delta=200 → home_strength=1.2, away_strength=0.8
    assert sig.home_strength == pytest.approx(1.2)
    assert sig.away_strength == pytest.approx(0.8)
    assert sig.raw_data["elo_delta"] == 200


@pytest.mark.asyncio
async def test_clip_at_400_elo_difference():
    # Anything beyond ±400 is clipped — protects against malformed YAML.
    sig = await EloStrengthFactor().compute(_ctx(2400, 1400))
    assert sig.raw_data["elo_delta"] == 400
    assert sig.home_strength == pytest.approx(1.4)
    assert sig.away_strength == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_lower_confidence_for_low_elo_teams():
    sig = await EloStrengthFactor().compute(_ctx(1280, 1250))
    # Both <1300 → looks like placeholder data → confidence drops.
    assert sig.confidence == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_missing_elo_marks_unavailable():
    ctx = FactorContext(
        match_id="m1",
        config={"teams": {"home": {"code": "X"}, "away": {"code": "Y"}}},
        home_code="X",
        away_code="Y",
        kickoff_utc=datetime(2026, 6, 20, 19, 0, tzinfo=timezone.utc),
    )
    sig = await EloStrengthFactor().compute(ctx)
    assert sig.available is False
    assert sig.home_strength == 1.0
    assert sig.confidence == 0.0
    assert sig.raw_data["reason"] == "missing_elo"


@pytest.mark.asyncio
async def test_factor_weight_can_be_overridden():
    f = EloStrengthFactor(weight=0.55)
    sig = await f.compute(_ctx(1900, 1800))
    assert sig.weight == pytest.approx(0.55)
