"""Factor registry — single place that decides which factors run, with which
weights, in which order.

The match_service builds the ensemble with `get_active_factors(settings)`. A
factor with weight 0 is silently skipped — that's the killswitch for
SquadAvailabilityFactor when Wikidata is mocked.
"""
from __future__ import annotations

from typing import Iterable

import structlog

from config.settings import Settings
from factors.base import Factor
from factors.elo_strength import EloStrengthFactor
from factors.fifa_ranking import FifaRankingFactor
from factors.form_factor import FormFactor
from factors.goal_efficiency import GoalEfficiencyFactor
from factors.head_to_head import HeadToHeadFactor
from factors.injury_news import InjuryNewsFactor
from factors.lineup_strength import LineupStrengthFactor
from factors.llm_sentiment import LlmSentimentFactor
from factors.market_odds import MarketOddsFactor
from factors.ml_blend import MlBlendFactor
from factors.ml_blend_lgbm import MlBlendLgbmFactor
from factors.momentum_drift import MomentumDriftFactor
from factors.network_strength import NetworkStrengthFactor
from factors.rest_travel import RestTravelFactor
from factors.sentiment_factor import SentimentFactor
from factors.squad_availability import SquadAvailabilityFactor
from factors.squad_value import SquadValueFactor
from factors.tournament_context import TournamentContextFactor
from factors.venue_altitude import VenueAltitudeFactor
from factors.weather import WeatherFactor

log = structlog.get_logger("factors.registry")


def get_active_factors(settings: Settings) -> list[Factor]:
    """Return every factor that has a non-zero target weight in settings.

    A factor with target weight 0 is silently skipped (the killswitch). The
    ensemble re-normalises whatever remains, so toggling a factor is a pure
    weight edit in settings/.env — no code change. Order matches the spec's
    narrative for log readability.
    """
    candidates: Iterable[tuple[Factor, float]] = (
        # Core spec factors.
        (EloStrengthFactor(weight=settings.factor_weight_elo), settings.factor_weight_elo),
        (FormFactor(weight=settings.factor_weight_form), settings.factor_weight_form),
        (HeadToHeadFactor(weight=settings.factor_weight_h2h), settings.factor_weight_h2h),
        (GoalEfficiencyFactor(weight=settings.factor_weight_goals), settings.factor_weight_goals),
        (TournamentContextFactor(weight=settings.factor_weight_context), settings.factor_weight_context),
        (SentimentFactor(weight=settings.factor_weight_sentiment), settings.factor_weight_sentiment),
        (SquadAvailabilityFactor(weight=settings.factor_weight_squad), settings.factor_weight_squad),
        # v3 additional factors (sport-science / market signals).
        (FifaRankingFactor(weight=settings.factor_weight_fifa_rank), settings.factor_weight_fifa_rank),
        (RestTravelFactor(weight=settings.factor_weight_rest_travel), settings.factor_weight_rest_travel),
        (VenueAltitudeFactor(weight=settings.factor_weight_altitude), settings.factor_weight_altitude),
        (MarketOddsFactor(weight=settings.factor_weight_market), settings.factor_weight_market),
        (WeatherFactor(weight=settings.factor_weight_weather), settings.factor_weight_weather),
        (InjuryNewsFactor(weight=settings.factor_weight_injury), settings.factor_weight_injury),
        (MomentumDriftFactor(weight=settings.factor_weight_momentum), settings.factor_weight_momentum),
        # Trained ML heads — dormant (weight 0) until artifacts are trained.
        (MlBlendFactor(weight=settings.factor_weight_ml), settings.factor_weight_ml),
        (MlBlendLgbmFactor(weight=settings.factor_weight_ml_lgbm), settings.factor_weight_ml_lgbm),
        # v3.3 — extended factor set (live scrapers + NVIDIA LLM + PageRank graph).
        (LlmSentimentFactor(weight=settings.factor_weight_llm_sentiment), settings.factor_weight_llm_sentiment),
        (LineupStrengthFactor(weight=settings.factor_weight_lineup), settings.factor_weight_lineup),
        (SquadValueFactor(weight=settings.factor_weight_squad_value), settings.factor_weight_squad_value),
        (NetworkStrengthFactor(weight=settings.factor_weight_network), settings.factor_weight_network),
    )

    active: list[Factor] = []
    for factor, w in candidates:
        if w <= 0:
            log.debug("factor_disabled", name=factor.name, weight=w)
            continue
        active.append(factor)

    log.debug("factors_active", names=[f.name for f in active], count=len(active))
    return active
