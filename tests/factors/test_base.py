"""Validation tests for FactorSignal and the Factor ABC."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from factors.base import Factor, FactorContext, FactorSignal


def test_factor_signal_basic_valid():
    sig = FactorSignal(
        name="elo",
        home_strength=1.15,
        away_strength=0.85,
        weight=0.3,
        confidence=0.8,
        available=True,
        source="yaml",
    )
    assert sig.home_strength == 1.15
    assert sig.away_strength == 0.85
    assert sig.cached_at is None


def test_factor_signal_clamps_extremes():
    # Anything outside [0.3, 2.5] gets clamped — a misbehaving connector
    # cannot dominate the ensemble.
    sig = FactorSignal(
        name="rogue",
        home_strength=2.9,
        away_strength=0.1,
        weight=0.2,
        confidence=0.5,
        available=True,
        source="test",
    )
    assert sig.home_strength == 2.5
    assert sig.away_strength == 0.3


def test_factor_signal_rejects_out_of_pydantic_range():
    # 3.5 is past the Pydantic Field ceiling (3.0) — fail before clamp.
    with pytest.raises(ValidationError):
        FactorSignal(
            name="x", home_strength=3.5, away_strength=1.0,
            weight=0.1, confidence=0.5, available=True, source="t",
        )


def test_factor_signal_weight_must_be_normalised():
    with pytest.raises(ValidationError):
        FactorSignal(
            name="x", home_strength=1.0, away_strength=1.0,
            weight=1.5, confidence=0.5, available=True, source="t",
        )


def test_factor_signal_confidence_bounds():
    with pytest.raises(ValidationError):
        FactorSignal(
            name="x", home_strength=1.0, away_strength=1.0,
            weight=0.3, confidence=1.1, available=True, source="t",
        )


def test_factor_neutral_helper_marks_unavailable():
    class DummyFactor(Factor):
        name = "dummy"
        default_weight = 0.25

        async def compute(self, ctx):
            return self._neutral(source="mock", reason="test")

    f = DummyFactor()
    sig = f._neutral(source="mock", reason="testing")
    assert sig.name == "dummy"
    assert sig.available is False
    assert sig.home_strength == 1.0
    assert sig.away_strength == 1.0
    assert sig.weight == 0.25
    assert sig.raw_data == {"reason": "testing"}


def test_factor_context_defaults_are_empty():
    ctx = FactorContext(
        match_id="m1",
        config={"teams": {}},
        home_code="GER",
        away_code="CIV",
        kickoff_utc=datetime(2026, 6, 20, 19, 0, tzinfo=timezone.utc),
    )
    assert ctx.historical_matches_home == []
    assert ctx.head_to_head == []
    assert ctx.fixture_meta is None
    assert ctx.sentiment_payload is None
