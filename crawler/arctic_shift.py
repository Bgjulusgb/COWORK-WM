"""Reddit scraper using Arctic Shift's public API — no credentials required.

Arctic Shift is the community-maintained successor to Pushshift, providing
historical Reddit data via a free public API.

API base: https://arctic-shift.photon-reddit.com/api
Key endpoints used:
  GET /api/posts/search
    Params: subreddit, query (requires subreddit), after (epoch int or ISO),
            before (epoch int), limit (max 100), sort (asc|desc)

Note: 'query' requires 'subreddit' to be set — cross-subreddit full-text
search is not supported. We run one request per subreddit.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from crawler.mock_reddit import FetchedPost

log = structlog.get_logger("crawler.arctic_shift")

_BASE = "https://arctic-shift.photon-reddit.com/api"
_HEADERS = {
    "User-Agent": "RedditOrakel/2.0 WM2026 research (educational project)",
    "Accept": "application/json",
}
_TIMEOUT = 20.0
_CONCURRENCY = 5
_GAP = 0.5   # polite delay; no documented rate limit


def _parse_arctic_post(d: dict, subreddit: str, tier: int) -> FetchedPost | None:
    """Parse one Arctic Shift post dict into FetchedPost."""
    title = (d.get("title") or "").strip()
    body = (d.get("selftext") or "").strip()
    if body in ("[removed]", "[deleted]", ""):
        body = ""
    if not (title or body):
        return None
    score = int(d.get("score", 0))
    if score < 0:
        return None

    created_raw = d.get("created_utc", 0)
    if isinstance(created_raw, str):
        try:
            created_raw = float(created_raw)
        except ValueError:
            created_raw = 0
    created_dt = datetime.fromtimestamp(float(created_raw), tz=timezone.utc) if created_raw else datetime.now(timezone.utc)

    reddit_id = d.get("id", "")
    import hashlib
    post_id = hashlib.md5(f"{subreddit}:{reddit_id}".encode()).hexdigest()[:12]

    return FetchedPost(
        post_id=post_id,
        subreddit=subreddit,
        tier=tier,
        title=title,
        body=body or title,
        score=score,
        upvote_ratio=float(d.get("upvote_ratio", 0.7)),
        num_comments=int(d.get("num_comments", 0)),
        created_utc=created_dt,
        author=d.get("author", "unknown"),
        is_comment=False,
        flair=d.get("link_flair_text"),
        source="arctic_shift",
        source_post_id=reddit_id or None,
    )


class ArcticShiftCrawler:
    """Fetches historical Reddit posts from Arctic Shift — no API key needed."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)

    async def _get_json(self, url: str, params: dict) -> dict | None:
        try:
            resp = await self._client.get(url, params=params)
            if resp.status_code != 200:
                log.warning("arctic_http_error", url=url, status=resp.status_code)
                return None
            data = resp.json()
            if data.get("error"):
                log.warning("arctic_api_error", error=data["error"], params=params)
                return None
            return data
        except Exception as exc:
            log.warning("arctic_fetch_failed", url=url, error=str(exc))
            return None

    async def _rate_limited(self, sem: asyncio.Semaphore, coro) -> list[FetchedPost]:
        async with sem:
            result = await coro
            await asyncio.sleep(_GAP)
            return result if result is not None else []

    async def _search_posts(
        self,
        subreddit: str,
        query: str | None = None,
        *,
        days_back: int = 21,
        limit: int = 100,
        tier: int = 1,
    ) -> list[FetchedPost]:
        """Search posts in one subreddit (most recent first, sorted descending)."""
        after_epoch = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
        params: dict[str, Any] = {
            "subreddit": subreddit,
            "after": after_epoch,
            "limit": min(limit, 100),
            "sort": "desc",
        }
        if query:
            params["query"] = query

        data = await self._get_json(f"{_BASE}/posts/search", params)
        if not data:
            return []

        posts = []
        for item in (data.get("data") or []):
            p = _parse_arctic_post(item, subreddit, tier)
            if p:
                posts.append(p)
        return posts

    async def crawl_for_match(self, config: dict) -> list[FetchedPost]:
        """Fetch historical posts for a match from Arctic Shift in parallel."""
        sources = config.get("reddit_sources", {})
        home_name = config["teams"]["home"]["name"]
        away_name = config["teams"]["away"]["name"]
        home_code = config["teams"]["home"]["code"]
        away_code = config["teams"]["away"]["code"]
        query_main = f"{home_name} {away_name}"

        sem = asyncio.Semaphore(_CONCURRENCY)
        tasks = []

        # Tier 1 — global subreddits with team name query
        for src in sources.get("tier1_global", []):
            sub = src["subreddit"]
            tasks.append(
                self._rate_limited(sem, self._search_posts(sub, query_main, days_back=21, limit=100, tier=1))
            )

        # Tier 2 — team subreddits (no query filter — get recent hot posts)
        for side_sources in sources.get("tier2_team_specific", {}).values():
            for src in side_sources:
                sub = src["subreddit"]
                tasks.append(
                    self._rate_limited(sem, self._search_posts(sub, days_back=14, limit=50, tier=2))
                )

        # Tier 3 — national subreddits with WC keyword
        for side, side_sources in sources.get("tier3_national_sentiment", {}).items():
            kw = f"world cup {home_code}" if side == "home" else f"world cup {away_code}"
            for src in side_sources:
                sub = src["subreddit"]
                tasks.append(
                    self._rate_limited(sem, self._search_posts(sub, kw, days_back=21, limit=30, tier=3))
                )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_posts: list[FetchedPost] = []
        for r in results:
            if isinstance(r, list):
                all_posts.extend(r)
            elif isinstance(r, Exception):
                log.warning("arctic_task_failed", error=str(r))

        # Deduplicate within Arctic Shift results
        seen: set[str] = set()
        unique: list[FetchedPost] = []
        for p in all_posts:
            key = p.source_post_id or f"{p.subreddit}:{p.post_id}"
            if key not in seen:
                seen.add(key)
                unique.append(p)

        log.info("arctic_crawl_done", tasks=len(tasks), posts=len(unique))
        return unique

    async def aclose(self) -> None:
        await self._client.aclose()
