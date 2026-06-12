"""Tests for the four v3.3 factors:

* LlmSentimentFactor — NVIDIA LLM aspect sentiment
* LineupStrengthFactor — FotMob/SofaScore confirmed XI vs season-avg
* SquadValueFactor — Transfermarkt log-ratio
* NetworkStrengthFactor — PageRank snapshot

Each factor is checked for:
  - correct directional tilt when its preferred side wins,
  - clean self-disable / neutral when data missing,
  - schema-clamped output (0.3..2.5 enforced by FactorSignal).
"""
from datetime import datetime, timezone

import pytest

from data_sources.schemas import LineupInfo, SquadValueInfo
from factors.base import FactorContext
from factors.lineup_strength import LineupStrengthFactor
from factors.llm_sentiment import LlmSentimentFactor
from factors.network_strength import NetworkStrengthFactor
from factors.squad_value import SquadValueFactor

_KICK = datetime(2026, 6, 20, 19, 0, tzinfo=timezone.utc)


def _ctx(**kw) -> FactorContext:
    base = dict(
        match_id="m1",
        config={"teams": {"home": {"code": "HOM"}, "away": {"code": "AWA"}}},
        home_code="HOM",
        away_code="AWA",
        kickoff_utc=_KICK,
    )
    base.update(kw)
    return FactorContext(**base)


# ── LlmSentimentFactor ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_sentiment_tilts_with_polarity():
    payload = {
        "model": "test-llm",
        "samples": 10,
        "home": {"polarity": 0.7, "intensity": 0.8, "confidence": 0.7,
                  "aspects": {"attack": 0.6, "defence": 0.0, "morale": 0.5}},
        "away": {"polarity": -0.3, "intensity": 0.5, "confidence": 0.6,
                  "aspects": {"attack": -0.4, "defence": -0.2, "morale": 0.0}},
    }
    sig = await LlmSentimentFactor(weight=0.1).compute(
        _ctx(sentiment_payload={"llm": payload})
    )
    assert sig.available is True
    assert sig.home_strength > 1.0
    assert sig.away_strength < 1.0
    assert 0.0 <= sig.confidence <= 1.0


@pytest.mark.asyncio
async def test_llm_sentiment_neutral_without_payload():
    sig = await LlmSentimentFactor(weight=0.1).compute(_ctx())
    assert sig.available is False
    assert sig.home_strength == 1.0
    assert sig.away_strength == 1.0


# ── LineupStrengthFactor ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lineup_strength_penalises_bench_promotions():
    home = LineupInfo(
        source="fotmob", code="HOM", is_confirmed=True,
        starters=[], starters_value_eur=300_000_000, season_avg_value_eur=300_000_000,
        bench_promotions=0,
    )
    away = LineupInfo(
        source="fotmob", code="AWA", is_confirmed=True,
        starters=[], starters_value_eur=240_000_000, season_avg_value_eur=300_000_000,
        bench_promotions=2,
    )
    sig = await LineupStrengthFactor(weight=0.05).compute(_ctx(lineup_home=home, lineup_away=away))
    assert sig.available is True
    assert sig.home_strength > sig.away_strength


@pytest.mark.asyncio
async def test_lineup_strength_neutral_when_missing():
    sig = await LineupStrengthFactor(weight=0.05).compute(_ctx())
    assert sig.available is False


# ── SquadValueFactor ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_squad_value_tilts_toward_richer_side():
    home = SquadValueInfo(source="tm", code="HOM",
                          total_value_eur=1_500_000_000, squad_size=26,
                          avg_value_eur=57_000_000, top11_value_eur=1_000_000_000)
    away = SquadValueInfo(source="tm", code="AWA",
                          total_value_eur=300_000_000, squad_size=26,
                          avg_value_eur=11_000_000, top11_value_eur=200_000_000)
    sig = await SquadValueFactor(weight=0.04).compute(
        _ctx(squad_value_home=home, squad_value_away=away)
    )
    assert sig.available is True
    assert sig.home_strength > 1.0
    assert sig.away_strength < 1.0
    assert sig.raw_data["log10_ratio"] > 0


@pytest.mark.asyncio
async def test_squad_value_neutral_when_missing():
    sig = await SquadValueFactor(weight=0.04).compute(_ctx())
    assert sig.available is False


# ── NetworkStrengthFactor ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_network_strength_neutral_without_snapshot(tmp_path, monkeypatch):
    # No artifact ⇒ factor self-disables.
    f = NetworkStrengthFactor(weight=0.05)
    sig = await f.compute(_ctx())
    assert sig.available is False
    assert sig.home_strength == 1.0


@pytest.mark.asyncio
async def test_network_strength_uses_in_memory_snapshot(monkeypatch):
    f = NetworkStrengthFactor(weight=0.05)
    # Bypass artifact loading.
    f._snapshot = {"HOM": 0.20, "AWA": 0.05, "OTHER": 0.10}
    f._loaded = True
    sig = await f.compute(_ctx())
    assert sig.available is True
    assert sig.home_strength > sig.away_strength
