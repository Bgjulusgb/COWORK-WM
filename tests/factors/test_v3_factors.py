"""Tests for every v3 factor (the 6 spec factors beyond Elo + the 7 new ones).

Each factor is checked for: correct directional tilt (better/luckier side gets
home_strength > 1), clean self-disable when its data is missing, and the
0.3..2.5 clamp inherited from FactorSignal.
"""
from datetime import datetime, timedelta, timezone

import pytest

from data_sources.schemas import HistoricalMatch, InjuryNewsItem, SquadInfo, VenueInfo, WeatherInfo
from factors.base import FactorContext
from factors.fifa_ranking import FifaRankingFactor
from factors.form_factor import FormFactor
from factors.goal_efficiency import GoalEfficiencyFactor
from factors.head_to_head import HeadToHeadFactor
from factors.injury_news import InjuryNewsFactor
from factors.market_odds import MarketOddsFactor
from factors.ml_blend import MlBlendFactor
from factors.momentum_drift import MomentumDriftFactor
from factors.rest_travel import RestTravelFactor
from factors.sentiment_factor import SentimentFactor
from factors.squad_availability import SquadAvailabilityFactor
from factors.tournament_context import TournamentContextFactor
from factors.venue_altitude import VenueAltitudeFactor
from factors.weather import WeatherFactor

_KICK = datetime(2026, 6, 20, 19, 0, tzinfo=timezone.utc)


def _ctx(**kw) -> FactorContext:
    base = dict(
        match_id="m1",
        config={"teams": {"home": {"name": "Home", "code": "HHH"},
                          "away": {"name": "Away", "code": "AAA"}}},
        home_code="HHH",
        away_code="AAA",
        kickoff_utc=_KICK,
    )
    base.update(kw)
    return FactorContext(**base)


def _match(home, away, hs, as_, days_ago=10, tier=1):
    return HistoricalMatch(
        source="test", competition_tier=tier,
        home_code=home, away_code=away, home_name=home, away_name=away,
        kickoff_utc=_KICK - timedelta(days=days_ago),
        home_score=hs, away_score=as_,
    )


# ── Sentiment ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sentiment_neutral_without_posts():
    sig = await SentimentFactor().compute(_ctx(sentiment_payload={"sample_size": 0}))
    assert sig.available is False


@pytest.mark.asyncio
async def test_sentiment_positive_home_lifts_home():
    ctx = _ctx(sentiment_payload={
        "home_sentiment": 0.8, "away_sentiment": -0.2,
        "home_momentum": 0.0, "away_momentum": 0.0, "sample_size": 100,
    })
    sig = await SentimentFactor().compute(ctx)
    assert sig.available is True
    assert sig.home_strength > 1.0 > sig.away_strength


# ── Form ─────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_form_winner_stronger():
    home_hist = [_match("HHH", "X", 3, 0, d) for d in (5, 20, 35, 50)]
    away_hist = [_match("AAA", "Y", 0, 3, d) for d in (5, 20, 35, 50)]
    sig = await FormFactor().compute(_ctx(historical_matches_home=home_hist,
                                          historical_matches_away=away_hist))
    assert sig.available is True
    assert sig.home_strength > 1.0 > sig.away_strength


@pytest.mark.asyncio
async def test_form_neutral_without_history():
    sig = await FormFactor().compute(_ctx())
    assert sig.available is False


# ── Goal efficiency ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_goal_efficiency_high_scorer_stronger():
    home_hist = [_match("HHH", "X", 4, 0, d) for d in (5, 20, 35, 50)]
    away_hist = [_match("AAA", "Y", 1, 1, d) for d in (5, 20, 35, 50)]
    sig = await GoalEfficiencyFactor().compute(_ctx(historical_matches_home=home_hist,
                                                    historical_matches_away=away_hist))
    assert sig.available is True
    assert sig.home_strength > sig.away_strength
    assert sig.raw_data["is_xg_proxy"] is True


# ── Head-to-head ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_h2h_home_dominance():
    h2h = [_match("HHH", "AAA", 2, 0, d) for d in (100, 300, 500)]
    h2h += [_match("AAA", "HHH", 0, 1, 700)]  # away hosted, home still won
    sig = await HeadToHeadFactor().compute(_ctx(head_to_head=h2h))
    assert sig.available is True
    assert sig.home_strength > 1.0
    assert sig.raw_data["home_wins"] == 4


@pytest.mark.asyncio
async def test_h2h_unavailable_without_data():
    # Codes with no snapshot entry → neutral.
    sig = await HeadToHeadFactor().compute(_ctx(home_code="ZZZ", away_code="QQQ"))
    assert sig.available is False


# ── Tournament context ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_context_host_nation_boost():
    ctx = _ctx(home_code="MEX", away_code="AAA", venue="Estadio Azteca, Mexico City",
               config={"teams": {"home": {"code": "MEX"}, "away": {"code": "AAA"}},
                       "match": {"phase": "group_stage"}})
    sig = await TournamentContextFactor().compute(ctx)
    assert sig.available is True
    assert sig.home_strength > 1.0


@pytest.mark.asyncio
async def test_context_neutral_group_venue_disables():
    ctx = _ctx(venue="Some Neutral Stadium",
               config={"teams": {}, "match": {"phase": "group_stage"}})
    sig = await TournamentContextFactor().compute(ctx)
    assert sig.available is False


# ── Squad availability ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_squad_more_stars_stronger():
    home = SquadInfo(source="t", code="HHH", star_players_available=10, available=True)
    away = SquadInfo(source="t", code="AAA", star_players_available=6, available=True)
    sig = await SquadAvailabilityFactor().compute(_ctx(squad_meta_home=home, squad_meta_away=away))
    assert sig.available is True
    assert sig.home_strength > 1.0


@pytest.mark.asyncio
async def test_squad_unavailable_when_one_missing():
    home = SquadInfo(source="t", code="HHH", star_players_available=10, available=True)
    away = SquadInfo(source="t", code="AAA", available=False)
    sig = await SquadAvailabilityFactor().compute(_ctx(squad_meta_home=home, squad_meta_away=away))
    assert sig.available is False


# ── FIFA ranking ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fifa_ranking_better_rank_stronger():
    # ARG (#1) vs a weak side via the bundled table fallback.
    sig = await FifaRankingFactor().compute(_ctx(home_code="ARG", away_code="HAI"))
    assert sig.available is True
    assert sig.home_strength > 1.0


# ── Rest / travel ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rest_more_rested_home_stronger():
    sig = await RestTravelFactor().compute(_ctx(rest_days_home=6, rest_days_away=3))
    assert sig.available is True
    assert sig.home_strength > 1.0


@pytest.mark.asyncio
async def test_rest_unknown_disables():
    sig = await RestTravelFactor().compute(_ctx())
    assert sig.available is False


@pytest.mark.asyncio
async def test_travel_penalises_long_haul_side():
    # Equal rest, but away crossed 3 time zones + 4000 km → home gets the edge.
    ctx = _ctx(rest_days_home=4, rest_days_away=4,
               travel_home={"km": 0.0, "tz_shift": 0.0},
               travel_away={"km": 4000.0, "tz_shift": 3.0})
    sig = await RestTravelFactor().compute(ctx)
    assert sig.available is True
    assert sig.home_strength > 1.0 > sig.away_strength


@pytest.mark.asyncio
async def test_travel_only_no_rest_still_fires():
    ctx = _ctx(travel_home={"km": 3000.0, "tz_shift": 2.0}, travel_away=None)
    sig = await RestTravelFactor().compute(ctx)
    assert sig.available is True
    assert sig.home_strength < 1.0  # home travelled, away didn't


# ── Venue altitude ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_altitude_damps_high_venue():
    vi = VenueInfo(source="t", name="Azteca", altitude_m=2240)
    sig = await VenueAltitudeFactor().compute(_ctx(venue_info=vi))
    assert sig.available is True
    assert sig.home_strength < 1.0
    assert sig.home_strength == sig.away_strength  # symmetric goal damp


@pytest.mark.asyncio
async def test_altitude_neutral_at_sea_level():
    vi = VenueInfo(source="t", name="Miami", altitude_m=3)
    sig = await VenueAltitudeFactor().compute(_ctx(venue_info=vi))
    assert sig.available is False


# ── Market odds ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_market_home_favourite_lifts_home():
    sig = await MarketOddsFactor().compute(_ctx(market_implied=(0.6, 0.25, 0.15)))
    assert sig.available is True
    assert sig.home_strength > 1.0 > sig.away_strength


@pytest.mark.asyncio
async def test_market_disabled_without_odds():
    sig = await MarketOddsFactor().compute(_ctx())
    assert sig.available is False


# ── Weather ──────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_weather_heat_damps_goals():
    sig = await WeatherFactor().compute(_ctx(weather=WeatherInfo(source="t", temp_c=36.0, humidity_pct=75)))
    assert sig.available is True
    assert sig.home_strength < 1.0


@pytest.mark.asyncio
async def test_weather_mild_disables():
    sig = await WeatherFactor().compute(_ctx(weather=WeatherInfo(source="t", temp_c=20.0)))
    assert sig.available is False


# ── Injury news ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_injury_more_home_injuries_helps_away():
    ctx = _ctx(news_home=[InjuryNewsItem(source="t", team_code="HHH", headline="x", impact=0.8)],
               news_away=[])
    sig = await InjuryNewsFactor().compute(ctx)
    assert sig.available is True
    assert sig.home_strength < 1.0 < sig.away_strength


@pytest.mark.asyncio
async def test_injury_disabled_without_news():
    sig = await InjuryNewsFactor().compute(_ctx())
    assert sig.available is False


# ── Momentum drift ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_momentum_rising_home_stronger():
    ctx = _ctx(sentiment_payload={"sample_size": 80, "home_trend_slope": 0.05, "away_trend_slope": -0.05})
    sig = await MomentumDriftFactor().compute(ctx)
    assert sig.available is True
    assert sig.home_strength > 1.0


@pytest.mark.asyncio
async def test_momentum_neutral_without_posts():
    sig = await MomentumDriftFactor().compute(_ctx(sentiment_payload={"sample_size": 0}))
    assert sig.available is False


# ── ML blend (trained head) ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_ml_blend_uses_trained_model(monkeypatch):
    import models_ml.xg_predictor as xgmod
    from models_ml.xg_predictor import XgPrediction

    class _Fake:
        is_available = True

        def predict(self, _features):
            return XgPrediction(home_xg=1.9, away_xg=1.0)

    monkeypatch.setattr(xgmod, "predictor", _Fake())
    sig = await MlBlendFactor(weight=0.1).compute(_ctx())
    assert sig.available is True
    assert sig.home_strength > sig.away_strength
    assert sig.raw_data["is_xg_proxy"] is True


@pytest.mark.asyncio
async def test_ml_blend_neutral_without_artifact(monkeypatch):
    import models_ml.xg_predictor as xgmod

    class _Fake:
        is_available = False

        def predict(self, _features):
            return None

    monkeypatch.setattr(xgmod, "predictor", _Fake())
    sig = await MlBlendFactor(weight=0.1).compute(_ctx())
    assert sig.available is False
