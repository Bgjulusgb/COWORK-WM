"""EXTEND-01: Twitter/X integration (skeleton).

Twitter v2 Search API. Requires a paid Bearer token now that the free tier
is gone, hence the entire integration is opt-in via TWITTER_BEARER_TOKEN +
ENABLE_TWITTER_CRAWLER.

The returned `FetchedPost` objects look identical to Reddit posts so the
downstream pipeline is reused unchanged.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Iterable

import httpx
import structlog

from config.settings import settings
from crawler.mock_reddit import FetchedPost

log = structlog.get_logger("crawler.twitter")

_SEARCH = "https://api.twitter.com/2/tweets/search/recent"
_TIMEOUT = 12.0


def _is_enabled() -> bool:
    return bool(settings.enable_twitter_crawler and settings.twitter_bearer_token)


def _hashtags(*codes: str) -> str:
    return " OR ".join(f"#{c}" for c in codes if c)


async def fetch_tweets(home_code: str, away_code: str, *, max_results: int = 50) -> list[FetchedPost]:
    if not _is_enabled():
        return []

    query = f"({_hashtags(home_code, away_code)}) (#WC2026 OR #WM2026 OR #WorldCup2026)"
    params = {
        "query": query,
        "max_results": min(100, max_results),
        "tweet.fields": "public_metrics,lang,created_at,author_id",
    }
    headers = {"Authorization": f"Bearer {settings.twitter_bearer_token}"}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.get(_SEARCH, params=params, headers=headers)
        except Exception as exc:
            log.warning("twitter_fetch_failed", error=str(exc))
            return []
        if resp.status_code == 429:
            retry = resp.headers.get("x-rate-limit-reset", "?")
            log.warning("twitter_rate_limit", reset=retry)
            return []
        if resp.status_code != 200:
            log.warning("twitter_http_error", status=resp.status_code)
            return []
        payload = resp.json()

    tweets = payload.get("data") or []
    out: list[FetchedPost] = []
    for t in tweets:
        metrics = t.get("public_metrics") or {}
        out.append(
            FetchedPost(
                post_id=f"tw:{t.get('id', '')}",
                subreddit="twitter",
                tier=4,
                title="",
                body=t.get("text") or "",
                score=int(metrics.get("like_count", 0)),
                upvote_ratio=1.0,
                num_comments=int(metrics.get("reply_count", 0)),
                created_utc=_parse_iso(t.get("created_at")),
                author=f"twitter:{t.get('author_id', 'unknown')}",
                is_comment=False,
                flair=t.get("lang"),
                source="twitter",
                source_post_id=t.get("id"),
            )
        )
    return out


def _parse_iso(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)
