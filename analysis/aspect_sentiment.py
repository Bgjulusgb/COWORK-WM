"""EXTEND-06: Aspect-Based Sentiment Analysis (ABSA).

Per-aspect breakdown of fan opinion: offensive, defensive, manager, fitness,
morale, tactics. The lookup is intentionally lexicon-based (not transformer-
based) so it works offline and adds <1ms per post.

Aspects are extracted via keyword presence; sentiment for an aspect is the
mean of the post's ensemble score restricted to posts where the aspect was
detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


ASPECTS: dict[str, set[str]] = {
    "offensive": {
        "attack", "attacking", "striker", "goal", "goals", "scorer", "scoring",
        "shot", "shots", "finishing", "finish", "xg", "forward", "winger",
        "creativity", "buildup",
    },
    "defensive": {
        "defense", "defence", "defender", "centerback", "centreback", "fullback",
        "clean sheet", "clearance", "tackle", "press", "pressing", "block",
        "shape", "compact", "leaky", "leaks",
    },
    "manager": {
        "manager", "coach", "boss", "gaffer", "tactician", "selection",
        "lineup", "starting xi", "rotation", "substitution",
    },
    "fitness": {
        "injury", "injured", "fitness", "tired", "exhausted", "recovery",
        "stamina", "matchfit", "rest", "fatigue",
    },
    "morale": {
        "confident", "confidence", "morale", "hyped", "hype", "belief",
        "doubt", "panic", "nervous", "anxious", "fearful",
    },
    "tactics": {
        "formation", "tactics", "tactical", "433", "352", "442", "high press",
        "low block", "counter", "possession", "structure", "system",
    },
}


@dataclass
class AspectBreakdown:
    aspect: str
    n: int
    mean_sentiment: float


@dataclass
class AspectReport:
    per_aspect: list[AspectBreakdown] = field(default_factory=list)

    def top(self, n: int = 3) -> list[AspectBreakdown]:
        return sorted(self.per_aspect, key=lambda a: -abs(a.mean_sentiment))[:n]


def _aspects_in(text: str) -> set[str]:
    if not text:
        return set()
    t = text.lower()
    return {asp for asp, kws in ASPECTS.items() if any(k in t for k in kws)}


def compute_aspect_report(posts: Iterable) -> AspectReport:
    sums: dict[str, float] = {a: 0.0 for a in ASPECTS}
    counts: dict[str, int] = {a: 0 for a in ASPECTS}
    for p in posts:
        # Posts here are the joined records from match_service._fetch_all_scored
        # — they don't expose body, so we fall back to processed_text via the
        # raw post when available. The caller passes RedditPost rows where
        # possible; for the in-memory _Joined view we read .body if present.
        text = getattr(p, "body", "") or getattr(p, "processed_text", "") or ""
        score = getattr(p, "ensemble_score", 0.0) or 0.0
        for asp in _aspects_in(text):
            sums[asp] += score
            counts[asp] += 1
    breakdown = [
        AspectBreakdown(aspect=a, n=counts[a], mean_sentiment=(sums[a] / counts[a]) if counts[a] else 0.0)
        for a in ASPECTS
    ]
    return AspectReport(per_aspect=breakdown)
