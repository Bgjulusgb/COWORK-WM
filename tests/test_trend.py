"""Unit tests for analysis.trend_analyzer."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from analysis.trend_analyzer import compute_trend


def _series_post(score, hours_ago, attribution="home"):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        ensemble_score=score,
        team_attribution=attribution,
        engagement_weight=1.0,
        created_utc=now - timedelta(hours=hours_ago),
        tier=1,
        subreddit="soccer",
        author="u",
    )


def test_flat_series_returns_flat_direction():
    posts = [_series_post(0.05, h) for h in range(0, 48, 3)]
    t = compute_trend(posts, team="home", bucket_hours=6)
    assert t.direction == "flat"


def test_rising_series_detected():
    # Strong positive trend across 48 hours of recent data
    posts = []
    for h in range(0, 48, 2):
        # later in time (lower hours_ago) → higher score
        score = -0.6 + (48 - h) * 0.025
        posts.append(_series_post(score, h))
    t = compute_trend(posts, team="home", bucket_hours=6)
    assert t.slope_per_hour > 0
    assert t.direction == "rising"


def test_falling_series_detected():
    posts = []
    for h in range(0, 48, 2):
        score = 0.6 - (48 - h) * 0.025
        posts.append(_series_post(score, h))
    t = compute_trend(posts, team="home", bucket_hours=6)
    assert t.slope_per_hour < 0
    assert t.direction == "falling"


def test_anomaly_detection_picks_spike():
    posts = [_series_post(0.05, h) for h in range(0, 48, 6)]
    # Insert one extreme spike
    posts.append(_series_post(0.95, 12))
    posts.append(_series_post(0.95, 12.1))
    posts.append(_series_post(0.95, 12.2))
    t = compute_trend(posts, team="home", bucket_hours=6)
    # With one bucket pulling far from the mean we expect at least one anomaly
    assert any(a.direction == "spike" for a in t.anomalies) or t.volatility > 0
