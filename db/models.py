from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, func,
)

from db.database import Base


class WM2026Match(Base):
    __tablename__ = "wm2026_matches"
    id = Column(String, primary_key=True)
    group = Column(String(2))
    phase = Column(String, index=True)
    home_team = Column(String(3))
    away_team = Column(String(3))
    home_name = Column(String)
    away_name = Column(String)
    home_flag = Column(String)
    away_flag = Column(String)
    kickoff_utc = Column(DateTime, index=True)
    venue = Column(String)
    status = Column(String, default="scheduled", index=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    config_path = Column(String)
    api_game_id = Column(String, nullable=True)
    # IMPROVE-06: timestamp of the last completed crawl so incremental
    # crawling can skip work that was already done.
    last_crawled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())


class RedditPost(Base):
    __tablename__ = "reddit_posts"
    id = Column(String, primary_key=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    subreddit = Column(String)
    tier = Column(Integer)
    post_id = Column(String, index=True)
    title = Column(Text)
    body = Column(Text)
    score = Column(Integer)
    upvote_ratio = Column(Float)
    num_comments = Column(Integer)
    created_utc = Column(DateTime)
    author = Column(String)
    flair = Column(String, nullable=True)
    is_comment = Column(Boolean, default=False)
    source = Column(String, default="reddit_json", nullable=False)
    raw_text = Column(Text)
    processed_text = Column(Text)
    detected_language = Column(String, default="en")
    translation_used = Column(Boolean, default=False)
    team_attribution = Column(String, nullable=True)
    crawled_at = Column(DateTime, default=func.now())

    __table_args__ = (
        # IMPROVE-11: speeds up time-windowed look-ups in advanced_metrics.
        Index("ix_reddit_posts_match_created", "match_id", "created_utc"),
    )


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String, ForeignKey("reddit_posts.id"), index=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    team = Column(String(8))
    vader_score = Column(Float)
    textblob_polarity = Column(Float)
    textblob_subjectivity = Column(Float)
    roberta_positive = Column(Float, nullable=True)
    roberta_neutral = Column(Float, nullable=True)
    roberta_negative = Column(Float, nullable=True)
    roberta_emotion = Column(String, nullable=True)
    ensemble_score = Column(Float)
    engagement_weight = Column(Float)
    source_language = Column(String, default="en")
    scored_at = Column(DateTime, default=func.now())


class MatchPrediction(Base):
    __tablename__ = "match_predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    generated_at = Column(DateTime, default=func.now())
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    confidence = Column(Float)
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_goals_expected = Column(Float)
    away_goals_expected = Column(Float)
    over_25_prob = Column(Float)
    over_15_prob = Column(Float)
    over_35_prob = Column(Float)
    btts_prob = Column(Float)
    top_scores = Column(JSON)
    recommended_bet = Column(String, nullable=True)
    bet_probability = Column(Float, nullable=True)
    features_snapshot = Column(JSON, nullable=True)
    actual_home_score = Column(Integer, nullable=True)
    actual_away_score = Column(Integer, nullable=True)
    prediction_correct = Column(Boolean, nullable=True)
    # MULTIFACTOR-05: per-factor breakdown for the v2 ensemble pipeline.
    # Nullable so rows from the legacy predictor stay valid. _add_missing_columns
    # in db.database picks this up on next init_db().
    factor_breakdown = Column(JSON, nullable=True)
    # v3.6 — Kalibrierte Vorhersagen + Pro-Modell-JSON. Nullable, weil:
    #  * neue Spalten -> _add_missing_columns triggert ALTER TABLE
    #  * Predictions vor erstem /calibrate -> raw == calibrated
    calibrated_home_win_prob = Column(Float, nullable=True)
    calibrated_draw_prob = Column(Float, nullable=True)
    calibrated_away_win_prob = Column(Float, nullable=True)
    platt_home_win_prob = Column(Float, nullable=True)
    platt_draw_prob = Column(Float, nullable=True)
    platt_away_win_prob = Column(Float, nullable=True)
    per_model_markets = Column(JSON, nullable=True)
    confidence_intervals = Column(JSON, nullable=True)
    # v3.7 (M2) — Bei jedem neuen Crawl bekommt die frische Row is_latest=True;
    # alle aelteren werden auf False demoted. Dashboards/Backtests koennen
    # damit ohne ORDER-BY-DESC-LIMIT-1 die jeweils aktuellste Vorhersage holen.
    is_latest = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        # IMPROVE-11: "latest prediction for match" query hits this directly.
        Index("ix_match_predictions_match_time", "match_id", "generated_at"),
        Index("ix_match_predictions_match_latest", "match_id", "is_latest"),
    )


class SentimentSnapshot(Base):
    __tablename__ = "sentiment_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    snapshot_time = Column(DateTime, default=func.now())
    hours_to_kickoff = Column(Float)
    home_sentiment = Column(Float)
    away_sentiment = Column(Float)
    home_momentum = Column(Float)
    away_momentum = Column(Float)
    home_post_velocity = Column(Float)
    away_post_velocity = Column(Float)
    total_posts_crawled = Column(Integer)
    # Advanced metrics (filled in by analysis.advanced_metrics + trend_analyzer)
    home_controversy = Column(Float, nullable=True)
    away_controversy = Column(Float, nullable=True)
    home_hype_ratio = Column(Float, nullable=True)
    away_hype_ratio = Column(Float, nullable=True)
    home_cope_ratio = Column(Float, nullable=True)
    away_cope_ratio = Column(Float, nullable=True)
    polarization = Column(Float, nullable=True)
    fan_balance = Column(Float, nullable=True)
    engagement_density = Column(Float, nullable=True)
    home_emotion = Column(String, nullable=True)
    away_emotion = Column(String, nullable=True)
    home_trend_slope = Column(Float, nullable=True)
    away_trend_slope = Column(Float, nullable=True)
    home_volatility = Column(Float, nullable=True)
    away_volatility = Column(Float, nullable=True)
    unique_authors = Column(Integer, nullable=True)
    # Heavy JSON blob (tier breakdown, subreddit influence, anomalies)
    advanced_payload = Column(JSON, nullable=True)

    __table_args__ = (
        # IMPROVE-11: dashboards hit "latest snapshot for match".
        Index("ix_sentiment_snapshots_match_time", "match_id", "snapshot_time"),
    )


class PostFlag(Base):
    """Per-post quality flags (bot, spam, low_quality). Audit trail for filters."""
    __tablename__ = "post_flags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String, nullable=True, index=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    flag = Column(String, index=True)   # "bot_author" | "low_quality" | "templated"
    detected_at = Column(DateTime, default=func.now())


class FactorSnapshot(Base):
    """MULTIFACTOR-06: one row per factor per prediction.

    Full audit trail of every signal that shaped a prediction. Lets the
    frontend explain the verdict and lets us back-test individual factors.
    """
    __tablename__ = "factor_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey("wm2026_matches.id"), index=True)
    prediction_id = Column(Integer, ForeignKey("match_predictions.id"), nullable=True, index=True)
    factor_name = Column(String, index=True)
    home_strength = Column(Float)
    away_strength = Column(Float)
    weight = Column(Float)                  # nominal weight from settings
    effective_weight = Column(Float)        # after re-normalisation across available factors
    confidence = Column(Float)
    available = Column(Boolean)
    source = Column(String)
    raw_data = Column(JSON, nullable=True)
    cached_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_factor_snapshots_match_factor", "match_id", "factor_name"),
    )


class DataSourceCache(Base):
    """MULTIFACTOR-07: persistent TTL cache for external connectors.

    In-memory TTLCache (utils/cache.py) is the fast path; this table is the
    slow-path backstop so restarts don't re-hit every connector. Caller is
    responsible for purging — we don't auto-evict on read.
    """
    __tablename__ = "data_source_cache"
    cache_key = Column(String, primary_key=True)   # sha256(connector|endpoint|sorted(params))
    connector = Column(String, index=True)
    endpoint = Column(String)
    payload = Column(JSON)
    fetched_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, index=True)


class TranslationCache(Base):
    """IMPROVE-10: persistent cache of Google Translate results.

    Key is sha256(source_lang + text). We never invalidate — translations are
    deterministic for a given input and Google's quality only improves over
    time, so the worst-case is stale-but-still-correct.
    """
    __tablename__ = "translation_cache"
    id = Column(String, primary_key=True)   # sha256 hex digest
    source_lang = Column(String(8), index=True)
    text_hash_prefix = Column(String(16))   # leading bytes of digest for cheap lookups
    translated = Column(Text)
    created_at = Column(DateTime, default=func.now())
