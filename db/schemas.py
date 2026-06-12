from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict


class MatchBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    group: str
    phase: str
    home_team: str
    away_team: str
    home_name: str
    away_name: str
    home_flag: Optional[str] = None
    away_flag: Optional[str] = None
    kickoff_utc: datetime
    venue: Optional[str] = None
    status: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None


class FactorSignalResponse(BaseModel):
    """MULTIFACTOR-11: serialised FactorSignal — mirror of factors/base.py.

    Lives in db/schemas.py so the API layer stays decoupled from the factor
    package (FastAPI response_model handling expects pydantic.BaseModel here).
    """
    model_config = ConfigDict(from_attributes=True)

    name: str
    home_strength: float
    away_strength: float
    weight: float
    effective_weight: float
    confidence: float
    available: bool
    source: str
    raw_data: Optional[dict] = None
    cached_at: Optional[datetime] = None
    is_xg_proxy: bool = False


class FactorBreakdownResponse(BaseModel):
    ensemble_confidence: float
    lambda_home_multiplier: float
    lambda_away_multiplier: float
    agreement: Optional[float] = None
    avg_factor_confidence: Optional[float] = None
    signals: List[FactorSignalResponse]
    notes: List[str] = []


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    match_id: str
    generated_at: datetime
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float
    home_xg: float
    away_xg: float
    home_goals_expected: float
    away_goals_expected: float
    over_25_prob: float
    over_15_prob: float
    over_35_prob: float
    btts_prob: float
    top_scores: List[dict]
    recommended_bet: Optional[str] = None
    bet_probability: Optional[float] = None
    # MULTIFACTOR-12: only present when USE_FACTOR_ENSEMBLE=true and the crawl
    # that produced this prediction ran on the new pipeline.
    factor_breakdown: Optional[FactorBreakdownResponse] = None
    # v3.6 — Kalibrierung + Pro-Modell-Output.
    calibrated_home_win_prob: Optional[float] = None
    calibrated_draw_prob: Optional[float] = None
    calibrated_away_win_prob: Optional[float] = None
    platt_home_win_prob: Optional[float] = None
    platt_draw_prob: Optional[float] = None
    platt_away_win_prob: Optional[float] = None
    per_model_markets: Optional[dict] = None
    confidence_intervals: Optional[dict] = None


class SentimentSnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    match_id: str
    snapshot_time: datetime
    home_sentiment: float
    away_sentiment: float
    home_momentum: float
    away_momentum: float
    home_post_velocity: float
    away_post_velocity: float
    total_posts_crawled: int
    home_controversy: Optional[float] = None
    away_controversy: Optional[float] = None
    home_hype_ratio: Optional[float] = None
    away_hype_ratio: Optional[float] = None
    home_cope_ratio: Optional[float] = None
    away_cope_ratio: Optional[float] = None
    polarization: Optional[float] = None
    fan_balance: Optional[float] = None
    engagement_density: Optional[float] = None
    home_emotion: Optional[str] = None
    away_emotion: Optional[str] = None
    home_trend_slope: Optional[float] = None
    away_trend_slope: Optional[float] = None
    home_volatility: Optional[float] = None
    away_volatility: Optional[float] = None
    unique_authors: Optional[int] = None


class TierBreakdown(BaseModel):
    n: int
    avg: float
    home_avg: float
    away_avg: float


class SubredditInfluence(BaseModel):
    subreddit: str
    post_count: int
    mean_sentiment: float
    weight: float
    influence: float


class AnomalyPoint(BaseModel):
    bucket_start: datetime
    delta: float
    z_score: float
    direction: str


class AdvancedMetricsResponse(BaseModel):
    match_id: str
    snapshot_time: datetime
    polarization: float
    fan_balance: float
    home_controversy: float
    away_controversy: float
    home_hype_ratio: float
    away_hype_ratio: float
    home_cope_ratio: float
    away_cope_ratio: float
    engagement_density: float
    home_emotion: str
    away_emotion: str
    home_trend_slope: float
    away_trend_slope: float
    home_volatility: float
    away_volatility: float
    unique_authors: int
    tier_breakdown: dict
    subreddit_influence: List[SubredditInfluence]
    anomalies: List[AnomalyPoint]


class RedditPostResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    subreddit: str
    tier: int
    title: Optional[str] = None
    body: Optional[str] = None
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    author: str
    team_attribution: Optional[str] = None
    ensemble_score: Optional[float] = None
    source: str = "reddit_json"


class TimelinePoint(BaseModel):
    hours_to_kickoff: float
    timestamp: datetime
    home_sentiment: float
    away_sentiment: float
    home_post_count: int
    away_post_count: int


class CrawlTriggerResponse(BaseModel):
    match_id: str
    posts_crawled: int
    posts_scored: int
    prediction_id: int
