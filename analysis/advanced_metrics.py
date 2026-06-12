"""Advanced sentiment & engagement metrics that extend the basic mean score.

These metrics turn a flat sentiment number into a multi-dimensional picture:

* `controversy_index`     — disagreement: high when posts are bimodally split
* `polarization`          — |home_share - away_share| of attributed posts
* `hype_vs_cope`          — share of very positive vs very negative posts
* `engagement_density`    — average engagement weight per post (popularity proxy)
* `tier_breakdown`        — per-tier sentiment so tier-1 (global) noise can be
                            separated from tier-2 (team-fan-base) skew
* `dominant_emotion`      — coarse label derived from score & subjectivity
* `subreddit_influence`   — weighted share each subreddit contributed to the score

All functions consume the same `_Joined` rows that `match_service` already
produces — they accept any object exposing `team_attribution`, `ensemble_score`,
`engagement_weight`, `created_utc`, `tier`, `subreddit`, `author`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


# Anything outside ±0.15 is considered a clearly polar opinion.
_NEUTRAL_BAND = 0.15
# Very strong opinions (hype/cope).
_STRONG_BAND = 0.50


@dataclass
class AdvancedMetrics:
    # Per-side
    home_controversy: float = 0.0
    away_controversy: float = 0.0
    home_hype_ratio: float = 0.0   # share of posts above +0.5
    away_hype_ratio: float = 0.0
    home_cope_ratio: float = 0.0   # share of posts below -0.5
    away_cope_ratio: float = 0.0

    # Cross-side
    polarization: float = 0.0      # |home_share - away_share|
    fan_balance: float = 0.0       # home_share - away_share, signed
    neutral_share: float = 0.0

    # Engagement
    engagement_density: float = 0.0       # avg engagement_weight per post
    top_decile_engagement: float = 0.0    # mean weight of top-10% loudest posts

    # Tier-resolved sentiment
    tier_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Subreddit influence (top-5 only to keep payload small)
    subreddit_influence: List[Dict[str, float]] = field(default_factory=list)

    # Heuristic emotion label per side
    home_emotion: str = "neutral"
    away_emotion: str = "neutral"

    # Sample sizes
    total_posts: int = 0
    home_posts: int = 0
    away_posts: int = 0
    neutral_posts: int = 0
    unique_authors: int = 0


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _controversy(scores: List[float]) -> float:
    """Bimodal disagreement index.

    Returns 0 if everyone agrees (low variance), grows with stdev of polar
    opinions while dampening for samples below 5 to avoid noise spikes.

    Range ~0..1.
    """
    polar = [s for s in scores if abs(s) >= _NEUTRAL_BAND]
    if len(polar) < 3:
        return 0.0
    std = pstdev(polar)
    # std maxes out around 1.0 for fully opposed -1/+1 mix.
    saturation = min(1.0, std)
    sample_dampen = min(1.0, len(polar) / 5.0)
    return float(saturation * sample_dampen)


def _hype_cope(scores: List[float]) -> Tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    n = len(scores)
    hype = sum(1 for s in scores if s >= _STRONG_BAND) / n
    cope = sum(1 for s in scores if s <= -_STRONG_BAND) / n
    return float(hype), float(cope)


def _emotion_label(mean_score: float, hype: float, cope: float, controversy: float) -> str:
    if controversy > 0.45:
        return "kontrovers"
    if mean_score >= 0.35 and hype >= 0.25:
        return "euphorisch"
    if mean_score >= 0.10:
        return "zuversichtlich"
    if mean_score <= -0.35 and cope >= 0.25:
        return "verzweifelt"
    if mean_score <= -0.10:
        return "skeptisch"
    return "neutral"


def _engagement_stats(posts) -> Tuple[float, float]:
    weights = [float(p.engagement_weight or 0.0) for p in posts]
    weights = [w for w in weights if w > 0]
    if not weights:
        return 0.0, 0.0
    avg = sum(weights) / len(weights)
    weights.sort(reverse=True)
    top_n = max(1, len(weights) // 10)
    top_decile = sum(weights[:top_n]) / top_n
    return float(avg), float(top_decile)


def _tier_breakdown(posts) -> Dict[str, Dict[str, float]]:
    by_tier: Dict[int, Dict[str, List[float]]] = {}
    for p in posts:
        t = getattr(p, "tier", None) or 0
        bucket = by_tier.setdefault(int(t), {"all": [], "home": [], "away": []})
        score = float(p.ensemble_score or 0.0)
        bucket["all"].append(score)
        if p.team_attribution == "home":
            bucket["home"].append(score)
        elif p.team_attribution == "away":
            bucket["away"].append(score)
    out: Dict[str, Dict[str, float]] = {}
    for tier, scores in sorted(by_tier.items()):
        key = f"tier{tier}"
        out[key] = {
            "n": len(scores["all"]),
            "avg": float(mean(scores["all"])) if scores["all"] else 0.0,
            "home_avg": float(mean(scores["home"])) if scores["home"] else 0.0,
            "away_avg": float(mean(scores["away"])) if scores["away"] else 0.0,
        }
    return out


def _subreddit_influence(posts, top_n: int = 5) -> List[Dict[str, float]]:
    """Aggregate per-subreddit weighted contribution to the mean.

    Returns the top `top_n` subreddits ranked by absolute influence
    (= |weighted_sum_of_scores|), each carrying its post count, mean
    sentiment, and total weight.
    """
    grouped: Dict[str, Dict[str, float]] = {}
    for p in posts:
        sub = getattr(p, "subreddit", None) or "unknown"
        w = float(p.engagement_weight or 0.0) or 1.0
        s = float(p.ensemble_score or 0.0)
        entry = grouped.setdefault(sub, {"n": 0, "weight": 0.0, "weighted_score": 0.0})
        entry["n"] += 1
        entry["weight"] += w
        entry["weighted_score"] += s * w

    ranked = []
    for sub, e in grouped.items():
        mean_s = _safe_div(e["weighted_score"], e["weight"])
        ranked.append(
            {
                "subreddit": sub,
                "post_count": int(e["n"]),
                "mean_sentiment": float(mean_s),
                "weight": float(e["weight"]),
                "influence": float(abs(e["weighted_score"])),
            }
        )
    ranked.sort(key=lambda x: x["influence"], reverse=True)
    return ranked[:top_n]


def compute_advanced(posts: Iterable) -> AdvancedMetrics:
    """Compute all advanced metrics over a list of joined post+sentiment rows.

    `posts` must be a sequence (iterated multiple times); pass a list.
    """
    posts = list(posts)
    if not posts:
        return AdvancedMetrics()

    home_scores: List[float] = []
    away_scores: List[float] = []
    authors: set = set()

    for p in posts:
        score = float(p.ensemble_score or 0.0)
        attr = p.team_attribution
        if attr == "home":
            home_scores.append(score)
        elif attr == "away":
            away_scores.append(score)
        else:
            # neutral posts contribute lightly to both sides
            home_scores.append(score)
            away_scores.append(score)
        auth = getattr(p, "author", None)
        if auth and auth != "unknown":
            authors.add(auth)

    n_total = len(posts)
    n_home = sum(1 for p in posts if p.team_attribution == "home")
    n_away = sum(1 for p in posts if p.team_attribution == "away")
    n_neutral = n_total - n_home - n_away

    home_hype, home_cope = _hype_cope(home_scores)
    away_hype, away_cope = _hype_cope(away_scores)
    home_contr = _controversy(home_scores)
    away_contr = _controversy(away_scores)

    home_share = _safe_div(n_home, n_total)
    away_share = _safe_div(n_away, n_total)
    neutral_share = _safe_div(n_neutral, n_total)
    polarization = abs(home_share - away_share)
    fan_balance = home_share - away_share

    avg_eng, top_eng = _engagement_stats(posts)

    home_mean = mean(home_scores) if home_scores else 0.0
    away_mean = mean(away_scores) if away_scores else 0.0
    home_emotion = _emotion_label(home_mean, home_hype, home_cope, home_contr)
    away_emotion = _emotion_label(away_mean, away_hype, away_cope, away_contr)

    return AdvancedMetrics(
        home_controversy=home_contr,
        away_controversy=away_contr,
        home_hype_ratio=home_hype,
        away_hype_ratio=away_hype,
        home_cope_ratio=home_cope,
        away_cope_ratio=away_cope,
        polarization=float(polarization),
        fan_balance=float(fan_balance),
        neutral_share=float(neutral_share),
        engagement_density=avg_eng,
        top_decile_engagement=top_eng,
        tier_breakdown=_tier_breakdown(posts),
        subreddit_influence=_subreddit_influence(posts),
        home_emotion=home_emotion,
        away_emotion=away_emotion,
        total_posts=n_total,
        home_posts=n_home,
        away_posts=n_away,
        neutral_posts=n_neutral,
        unique_authors=len(authors),
    )
