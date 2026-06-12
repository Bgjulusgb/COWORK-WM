"""Unit tests for analysis.advanced_metrics."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from analysis.advanced_metrics import compute_advanced


def _post(*, score, attribution="home", tier=1, sub="r1",
          author=None, weight=1.0, hours_ago=1.0):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        ensemble_score=score,
        team_attribution=attribution,
        engagement_weight=weight,
        created_utc=now - timedelta(hours=hours_ago),
        tier=tier,
        subreddit=sub,
        author=author or f"u_{int(score * 100)}",
    )


def test_empty_input_returns_defaults():
    m = compute_advanced([])
    assert m.total_posts == 0
    assert m.home_emotion == "neutral"
    assert m.subreddit_influence == []


def test_polarization_when_one_side_dominates():
    posts = [_post(score=0.5, attribution="home") for _ in range(8)]
    posts += [_post(score=-0.3, attribution="away") for _ in range(2)]
    m = compute_advanced(posts)
    assert m.home_posts == 8
    assert m.away_posts == 2
    assert m.polarization > 0.5
    assert m.fan_balance > 0
    assert m.home_emotion in ("euphorisch", "zuversichtlich")


def test_controversy_high_when_polar_opposites():
    posts = [_post(score=0.9, attribution="home") for _ in range(6)]
    posts += [_post(score=-0.9, attribution="home") for _ in range(6)]
    m = compute_advanced(posts)
    assert m.home_controversy > 0.5
    assert m.home_emotion == "kontrovers"


def test_hype_and_cope_ratios():
    posts = [_post(score=0.7, attribution="home") for _ in range(5)]
    posts += [_post(score=-0.7, attribution="home") for _ in range(5)]
    posts += [_post(score=0.0, attribution="home") for _ in range(10)]
    m = compute_advanced(posts)
    assert abs(m.home_hype_ratio - 0.25) < 0.01
    assert abs(m.home_cope_ratio - 0.25) < 0.01


def test_tier_breakdown_groups_by_tier():
    posts = [
        _post(score=0.5, tier=1),
        _post(score=0.4, tier=1),
        _post(score=-0.3, tier=2),
    ]
    m = compute_advanced(posts)
    assert "tier1" in m.tier_breakdown
    assert "tier2" in m.tier_breakdown
    assert m.tier_breakdown["tier1"]["n"] == 2
    assert m.tier_breakdown["tier2"]["n"] == 1


def test_subreddit_influence_ranks_by_weighted_signal():
    posts = [
        _post(score=0.8, sub="soccer", weight=10),
        _post(score=0.7, sub="soccer", weight=10),
        _post(score=0.1, sub="random", weight=1),
    ]
    m = compute_advanced(posts)
    assert m.subreddit_influence[0]["subreddit"] == "soccer"


def test_unique_authors_dedup():
    posts = [
        _post(score=0.1, author="alice"),
        _post(score=0.2, author="alice"),
        _post(score=0.3, author="bob"),
    ]
    m = compute_advanced(posts)
    assert m.unique_authors == 2
