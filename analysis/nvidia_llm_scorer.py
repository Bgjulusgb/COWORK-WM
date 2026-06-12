"""High-level wrapper around the NVIDIA-LLM connector.

Sits alongside :mod:`analysis.roberta_scorer` and is called by ``match_service``
when ``settings.use_nvidia_llm`` is on. Picks the top-N posts by engagement
weight from each Reddit tier (so we stay inside the LLM token budget) and
returns the aspect-sentiment payload the LlmSentimentFactor consumes.

Result format::

    {
        "model": "meta/llama-3.3-70b-instruct",
        "samples": int,
        "home": {polarity, intensity, confidence, aspects: {attack, defence, morale}},
        "away": {...},
    }

The mock path is deterministic, so tests that exercise the LlmSentimentFactor
do not need network access.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

import structlog

from config.settings import settings
from data_sources.nvidia_llm import NvidiaLlmConnector

log = structlog.get_logger("analysis.nvidia_llm_scorer")


def _post_text(p: Any) -> str:
    """Defensive: posts may arrive as dataclasses, dicts or BaseModels."""
    if hasattr(p, "model_dump"):
        try:
            d = p.model_dump()
        except Exception:
            d = {}
    elif is_dataclass(p):
        d = asdict(p)
    elif isinstance(p, dict):
        d = p
    else:
        return str(p)
    title = (d.get("title") or "").strip()
    body = (d.get("body") or "").strip()
    return (title + " " + body).strip()


def _engagement(p: Any) -> float:
    """Higher = more representative; used to rank within a tier."""
    if hasattr(p, "model_dump"):
        d = p.model_dump()
    elif is_dataclass(p):
        d = asdict(p)
    elif isinstance(p, dict):
        d = p
    else:
        return 0.0
    score = float(d.get("score", 0) or 0)
    num_comments = float(d.get("num_comments", 0) or 0)
    return score + 2.0 * num_comments


def _select_top(posts: Iterable[Any], k: int) -> list[str]:
    pool: list[tuple[float, str]] = []
    for p in posts:
        text = _post_text(p)
        if not text or len(text) < 10:
            continue
        pool.append((_engagement(p), text))
    pool.sort(key=lambda t: t[0], reverse=True)
    return [t[1] for t in pool[:k]]


async def score_match(
    home_code: str,
    away_code: str,
    posts: Iterable[Any],
    *,
    connector: NvidiaLlmConnector | None = None,
) -> dict | None:
    """Score one match via the NVIDIA LLM. Returns the payload or ``None``
    when neither live nor mock can produce something usable.
    """
    if not settings.use_nvidia_llm and not settings.nvidia_api_key:
        # Behave as if the feature is off; the factor stays neutral.
        return None
    cap = max(1, settings.llm_max_posts_per_tier)
    texts = _select_top(posts, cap * 3)
    if not texts:
        return None

    client = connector or NvidiaLlmConnector()
    res = await client.score_sentiment(texts, home_code, away_code)
    if not res.ok and res.data is None:
        log.warning("nvidia_llm_no_data", home=home_code, away=away_code, mode=res.mode)
        return None
    return res.data


__all__ = ["score_match"]
