from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


# IMPROVE-05: Tier multiplier applied on top of the post's engagement weight.
# Tier-1 (worldcup/soccer) is the baseline; tier-2 (national/team subs) gets a
# small bump because fans there are domain experts; tier-3 (broad national
# subreddits) is dialled down because most posts there aren't soccer-related.
_TIER_MULTIPLIER = {1: 1.0, 2: 1.5, 3: 0.8}


def tier_multiplier(tier: Optional[int]) -> float:
    if tier is None:
        return 1.0
    return _TIER_MULTIPLIER.get(int(tier), 1.0)


def temporal_weight(created_utc: datetime, kickoff_utc: Optional[datetime] = None) -> float:
    """Posts closer to kickoff get higher weight. Linear decay over 14 days.

    Age 0 days → 1.0x. Age 7 days → 0.75x. Age 14 days → 0.5x (floor).
    Posts older than 14 days are not discarded, just down-weighted to 0.5x.
    """
    reference = kickoff_utc or datetime.now(timezone.utc)
    age_days = max(0.0, (reference - created_utc).total_seconds()) / 86400.0
    return max(0.5, 1.0 - age_days / 28.0)


def weighted_sentiment(
    posts,
    *,
    team: str,
    score_field: str = "ensemble_score",
    weight_field: str = "engagement_weight",
    kickoff_utc: Optional[datetime] = None,
) -> Tuple[float, int]:
    relevant = [p for p in posts if p.team_attribution in (team, "neutral")]
    if not relevant:
        return 0.0, 0
    weight_sum = 0.0
    score_sum = 0.0
    for p in relevant:
        eng_w = getattr(p, weight_field) or 1.0
        temp_w = temporal_weight(p.created_utc, kickoff_utc)
        tier_w = tier_multiplier(getattr(p, "tier", None))
        w = eng_w * temp_w * tier_w
        weight_sum += w
        score_sum += (getattr(p, score_field) or 0.0) * w
    if weight_sum == 0.0:
        weight_sum = float(len(relevant))
    return score_sum / weight_sum, len(relevant)


def momentum(posts, *, team: str, window_hours: int = 6, kickoff_utc: Optional[datetime] = None) -> float:
    now = datetime.now(timezone.utc)
    recent_cut = now - timedelta(hours=window_hours)
    prior_cut = now - timedelta(hours=2 * window_hours)

    recent = [p for p in posts if p.created_utc >= recent_cut]
    prior = [p for p in posts if prior_cut <= p.created_utc < recent_cut]

    recent_score, _ = weighted_sentiment(recent, team=team, kickoff_utc=kickoff_utc)
    prior_score, _ = weighted_sentiment(prior, team=team, kickoff_utc=kickoff_utc)
    return recent_score - prior_score


def post_velocity(posts, *, team: str, hours: int = 6) -> float:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    relevant = [p for p in posts if p.created_utc >= cutoff and p.team_attribution in (team, "neutral")]
    return len(relevant) / hours
