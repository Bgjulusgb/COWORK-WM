"""Mock Reddit crawler producing realistic-looking posts for offline development.

Replace with PRAW/asyncpraw client once credentials are available — the public
interface (`crawl`) stays the same, downstream code doesn't change.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


@dataclass
class FetchedPost:
    post_id: str
    subreddit: str
    tier: int
    title: str
    body: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    author: str
    is_comment: bool = False
    flair: str | None = None
    source: str = "reddit_json"          # "reddit_json" | "arctic_shift"
    source_post_id: str | None = None    # original Reddit base36 ID for cross-source dedup


_HOME_POSITIVE = [
    "{home} are looking absolutely class right now, {star} is on fire",
    "Three Lions defense has been clinical all tournament, clean sheet incoming",
    "{home} dominate possession, {star} is a worldie waiting to happen",
    "Honestly {home} look unstoppable, expecting a brace from {star}",
    "Quality from {home} in training videos, atmosphere is great",
    "{home} fans are hyped, the squad depth this year is excellent",
]
_HOME_NEGATIVE = [
    "{home} bottled it again in the warmup, garbage form lately",
    "Not feeling great about {home} chances, defense looks shaky",
    "{star} is unlucky with the injury, terrible timing for {home}",
    "{home} have been trash in recent friendlies, expect disappointment",
    "The {home} squad selection is questionable, bottled the prep",
]
_HOME_NEUTRAL = [
    "What do you think about {home}'s lineup for tomorrow?",
    "Anyone got the {home} starting XI prediction?",
    "{home} pre-match thread, share your thoughts",
]

_AWAY_POSITIVE = [
    "{away} have looked sharp, {star} is in incredible form",
    "Vatreni are coming with their A-game, quality midfield is class",
    "{away} defense has been clinical, expecting a clean sheet",
    "{star} pulled a brace last week, banging form for {away}",
]
_AWAY_NEGATIVE = [
    "{away} are not looking good, {star} has been disappointing",
    "Honestly {away} bottled their qualifiers, garbage tournament prep",
    "{away} chances look terrible, defense is leaking goals",
    "{star} unlucky with form, this is a bad time for {away}",
]
_AWAY_NEUTRAL = [
    "{away} match thread, what's everyone thinking?",
    "Predictions for {away} starting XI?",
    "Pre-game analysis for {away} please",
]


def _team_pool(team_name: str, kind: str) -> List[str]:
    if kind == "home_positive":
        return _HOME_POSITIVE
    if kind == "home_negative":
        return _HOME_NEGATIVE
    if kind == "home_neutral":
        return _HOME_NEUTRAL
    if kind == "away_positive":
        return _AWAY_POSITIVE
    if kind == "away_negative":
        return _AWAY_NEGATIVE
    return _AWAY_NEUTRAL


def _make_post(
    rng: random.Random,
    subreddit: str,
    tier: int,
    template: str,
    team_name: str,
    star: str,
    hours_ago: float,
    *,
    sentiment_hint: str,
) -> FetchedPost:
    text = template.format(home=team_name, away=team_name, star=star)
    # BUG-14: use the full MD5 (128 bit) instead of truncating to 10 chars
    # (~48 bit, collision-prone). Stays deterministic for the same seed so
    # the idempotent-rerun test still passes.
    digest = "mock_" + hashlib.md5(
        f"{subreddit}|{text}|{hours_ago:.6f}".encode("utf-8")
    ).hexdigest()

    score = rng.randint(5, 800) if sentiment_hint != "neutral" else rng.randint(2, 80)
    num_comments = max(0, int(score / rng.uniform(3, 12)))
    upvote_ratio = round(rng.uniform(0.65, 0.97), 2)
    author = f"user_{rng.randint(1000, 9999)}"
    created = datetime.now(timezone.utc) - timedelta(hours=hours_ago)

    is_comment = rng.random() < 0.4
    return FetchedPost(
        post_id=digest,
        subreddit=subreddit,
        tier=tier,
        title="" if is_comment else text[:90],
        body=text,
        score=score,
        upvote_ratio=upvote_ratio,
        num_comments=num_comments,
        created_utc=created,
        author=author,
        is_comment=is_comment,
    )


class MockRedditCrawler:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    async def crawl(self, match_config: Dict[str, Any]) -> List[FetchedPost]:
        teams = match_config["teams"]
        home_name = teams["home"]["name"]
        away_name = teams["away"]["name"]
        home_star = self._pick_star(home_name)
        away_star = self._pick_star(away_name)

        sources = match_config["reddit_sources"]
        subreddits: List[tuple[str, int]] = []
        for entry in sources.get("tier1_global", []):
            subreddits.append((entry["subreddit"], 1))
        for entry in sources.get("tier2_team_specific", {}).get("home", []):
            subreddits.append((entry["subreddit"], 2))
        for entry in sources.get("tier2_team_specific", {}).get("away", []):
            subreddits.append((entry["subreddit"], 2))
        for entry in sources.get("tier3_national_sentiment", {}).get("home", []):
            subreddits.append((entry["subreddit"], 3))
        for entry in sources.get("tier3_national_sentiment", {}).get("away", []):
            subreddits.append((entry["subreddit"], 3))

        posts: List[FetchedPost] = []
        for subreddit, tier in subreddits:
            n = self.rng.randint(15, 40)
            for _ in range(n):
                bias = self.rng.choice(["home", "away", "neutral"])
                tone = self.rng.choices(
                    ["positive", "negative", "neutral"],
                    weights=self._tone_weights(bias, team_subreddit=subreddit, home=home_name, away=away_name),
                    k=1,
                )[0]
                if bias == "home":
                    pool = _team_pool(home_name, f"home_{tone}")
                    team_name, star = home_name, home_star
                elif bias == "away":
                    pool = _team_pool(away_name, f"away_{tone}")
                    team_name, star = away_name, away_star
                else:
                    pool = _team_pool(home_name, "home_neutral") + _team_pool(away_name, "away_neutral")
                    team_name, star = (home_name, home_star) if self.rng.random() < 0.5 else (away_name, away_star)
                template = self.rng.choice(pool)
                hours_ago = self.rng.uniform(0.5, 72.0)
                posts.append(
                    _make_post(
                        self.rng, subreddit, tier, template, team_name, star, hours_ago,
                        sentiment_hint=tone,
                    )
                )
        return posts

    @staticmethod
    def _pick_star(team: str) -> str:
        stars = {
            "England": "Bellingham",
            "Croatia": "Modric",
        }
        return stars.get(team, team)

    @staticmethod
    def _tone_weights(bias: str, *, team_subreddit: str, home: str, away: str):
        # Team-specific subreddits skew positive toward their own team
        sub = team_subreddit.lower()
        if bias == "home" and home.lower() in sub:
            return [0.55, 0.20, 0.25]
        if bias == "away" and away.lower() in sub:
            return [0.55, 0.20, 0.25]
        return [0.40, 0.35, 0.25]
