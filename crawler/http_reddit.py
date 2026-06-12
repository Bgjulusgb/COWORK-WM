"""Credentialless Reddit scraper using Reddit's public JSON API.

Reddit serves every page as JSON when you append .json — no OAuth needed.
Rate limit handling:
  - bounded semaphore (8 concurrent)
  - exponential backoff on 429 (1s -> 2s -> 4s -> 8s, then give up)
  - per-subreddit circuit breaker after 3 consecutive 429/403 responses
  - retry counter exposed via the `last_run_stats` dict for the API layer

Usage: set USE_MOCK_CRAWLER=false in .env to switch from MockRedditCrawler.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from crawler.mock_reddit import FetchedPost

log = structlog.get_logger("crawler.http_reddit")

_BASE = "https://old.reddit.com"
# old.reddit.com still serves JSON for anonymous clients in 2026 while
# www.reddit.com aggressively 403s scrapers (even with a legitimate UA).
# Browser-shaped headers reduce false-positive anti-bot trips further.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}
_TIMEOUT = 15.0
_CONCURRENCY = 4   # smaller burst → fewer 429/403; old.reddit is happier with this
_GAP = 1.0         # seconds to sleep after each request (inside semaphore)
_RATE_LIMIT_BACKOFF = (1, 2, 4, 8)
_CIRCUIT_BREAKER_THRESHOLD = 3


def _make_id(subreddit: str, reddit_id: str) -> str:
    """Use the natural Reddit base36 id — already collision-free."""
    return f"{subreddit}:{reddit_id}"


def _parse_post(child: dict, subreddit: str, tier: int,
                *, min_score: int = 0) -> FetchedPost | None:
    d = child.get("data", {})
    body = (d.get("selftext") or d.get("body") or "").strip()
    title = d.get("title") or d.get("link_title") or ""
    if not (body or title):
        return None
    score = int(d.get("score", 0))
    # IMPROVE-07: respect min_post_score from YAML, not just `score < 0`.
    if score < max(0, min_score):
        return None
    created = d.get("created_utc", 0)
    reddit_id = d.get("id", "unknown")
    return FetchedPost(
        post_id=_make_id(subreddit, reddit_id),
        subreddit=subreddit,
        tier=tier,
        title=title,
        body=body or title,
        score=score,
        upvote_ratio=float(d.get("upvote_ratio", 0.5)),
        num_comments=int(d.get("num_comments", 0)),
        created_utc=datetime.fromtimestamp(created, tz=timezone.utc) if created else datetime.now(timezone.utc),
        author=d.get("author", "unknown"),
        is_comment="link_title" in d,
        flair=d.get("link_flair_text") or d.get("author_flair_text"),
        source="reddit_json",
        source_post_id=reddit_id,
    )


def _deduplicate(posts: list[FetchedPost]) -> list[FetchedPost]:
    seen: set[str] = set()
    unique: list[FetchedPost] = []
    for p in posts:
        key = p.source_post_id or f"{p.subreddit}:{p.post_id}"
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


class CircuitBreaker:
    """Trip after N consecutive failures on a single subreddit."""

    def __init__(self, threshold: int = _CIRCUIT_BREAKER_THRESHOLD) -> None:
        self._failures: dict[str, int] = defaultdict(int)
        self._tripped: set[str] = set()
        self.threshold = threshold

    def is_tripped(self, subreddit: str) -> bool:
        return subreddit in self._tripped

    def record_failure(self, subreddit: str) -> None:
        self._failures[subreddit] += 1
        if self._failures[subreddit] >= self.threshold:
            self._tripped.add(subreddit)
            log.warning("circuit_breaker_tripped", subreddit=subreddit, failures=self._failures[subreddit])

    def record_success(self, subreddit: str) -> None:
        self._failures[subreddit] = 0
        self._tripped.discard(subreddit)


class HttpRedditCrawler:
    """Drop-in replacement for MockRedditCrawler — uses Reddit public JSON API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        self._breaker = CircuitBreaker()
        self.last_run_stats: dict[str, int] = {"requests": 0, "rate_limited": 0, "blocked": 0, "errors": 0}

    async def _get_json(self, url: str, subreddit: str, params: dict | None = None) -> dict | None:
        if self._breaker.is_tripped(subreddit):
            return None
        self.last_run_stats["requests"] += 1
        for attempt, backoff in enumerate(_RATE_LIMIT_BACKOFF):
            try:
                resp = await self._client.get(url, params=params)
            except Exception as exc:
                log.warning("reddit_fetch_failed", url=url, error=str(exc))
                self.last_run_stats["errors"] += 1
                self._breaker.record_failure(subreddit)
                return None

            if resp.status_code == 200:
                self._breaker.record_success(subreddit)
                return resp.json()

            if resp.status_code == 429:
                self.last_run_stats["rate_limited"] += 1
                log.warning("reddit_rate_limit", url=url, attempt=attempt + 1, backoff_s=backoff)
                await asyncio.sleep(backoff)
                self._breaker.record_failure(subreddit)
                continue
            if resp.status_code == 403:
                self.last_run_stats["blocked"] += 1
                log.warning("reddit_blocked", url=url)
                self._breaker.record_failure(subreddit)
                return None
            log.warning("reddit_http_error", url=url, status=resp.status_code)
            self.last_run_stats["errors"] += 1
            self._breaker.record_failure(subreddit)
            return None
        # Exhausted retries.
        return None

    async def _rate_limited(self, sem: asyncio.Semaphore, coro) -> list[FetchedPost]:
        async with sem:
            result = await coro
            await asyncio.sleep(_GAP)
            return result if result is not None else []

    async def _fetch_search(self, subreddit: str, query: str,
                            limit: int = 100, tier: int = 1,
                            pages: int = 2, *, min_score: int = 0) -> list[FetchedPost]:
        url = f"{_BASE}/r/{subreddit}/search.json"
        all_posts: list[FetchedPost] = []
        after: str | None = None
        for _ in range(pages):
            params = {"q": query, "sort": "new", "t": "month", "limit": min(limit, 100), "restrict_sr": "true"}
            if after:
                params["after"] = after
            data = await self._get_json(url, subreddit, params)
            if not data:
                break
            for child in data.get("data", {}).get("children", []):
                p = _parse_post(child, subreddit, tier, min_score=min_score)
                if p:
                    all_posts.append(p)
            after = data.get("data", {}).get("after")
            if not after:
                break
        return all_posts

    async def _fetch_hot(self, subreddit: str, limit: int = 50, tier: int = 1,
                         *, min_score: int = 0) -> list[FetchedPost]:
        url = f"{_BASE}/r/{subreddit}/hot.json"
        data = await self._get_json(url, subreddit, {"limit": limit})
        if not data:
            return []
        posts = []
        for child in data.get("data", {}).get("children", []):
            p = _parse_post(child, subreddit, tier, min_score=min_score)
            if p:
                posts.append(p)
        return posts

    async def _fetch_comments_for_top_posts(
        self,
        subreddit: str,
        tier: int,
        posts: list[FetchedPost],
        *,
        min_post_score: int = 50,
        depth: int = 2,
        max_posts: int = 5,
        min_score: int = 0,
    ) -> list[FetchedPost]:
        """IMPROVE-08: harvest comments from hot tier-2 posts.

        For each high-score post (score >= `min_post_score`), fetch top-level
        comments up to `depth`. We cap at `max_posts` to keep request volume
        bounded.
        """
        comments: list[FetchedPost] = []
        candidates = [p for p in posts if p.score >= min_post_score and not p.is_comment][:max_posts]
        for parent in candidates:
            if parent.source_post_id is None:
                continue
            url = f"{_BASE}/r/{subreddit}/comments/{parent.source_post_id}.json"
            data = await self._get_json(url, subreddit, {"depth": depth, "limit": 50})
            if not data or not isinstance(data, list) or len(data) < 2:
                continue
            for child in data[1].get("data", {}).get("children", []):
                if child.get("kind") != "t1":
                    continue
                p = _parse_post(child, subreddit, tier, min_score=min_score)
                if p:
                    comments.append(p)
        return comments

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _src_min_score(src: dict) -> int:
        try:
            return int(src.get("min_post_score", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _src_include_comments(src: dict) -> bool:
        return bool(src.get("include_comments", False))

    @staticmethod
    def _src_comment_depth(src: dict) -> int:
        try:
            return int(src.get("comment_depth", 2) or 2)
        except (TypeError, ValueError):
            return 2

    async def crawl(self, config: dict) -> list[FetchedPost]:
        self.last_run_stats = {"requests": 0, "rate_limited": 0, "blocked": 0, "errors": 0}
        sources = config.get("reddit_sources", {})
        home_name = config["teams"]["home"]["name"]
        away_name = config["teams"]["away"]["name"]
        home_code = config["teams"]["home"]["code"]
        away_code = config["teams"]["away"]["code"]

        sem = asyncio.Semaphore(_CONCURRENCY)
        # Phase 1: queue main fetches. We collect tier-2 hot results separately
        # so we can use them as comment seeds in phase 2.
        primary_tasks: list[tuple[str, asyncio.Future]] = []
        tier2_with_comments: list[tuple[str, int, int]] = []  # (subreddit, tier, min_score)

        for src in sources.get("tier1_global", []):
            sub = src["subreddit"]
            keywords = src.get("search_keywords", [home_name, away_name])
            query = " OR ".join(keywords[:3])
            min_score = self._src_min_score(src)
            primary_tasks.append((sub, asyncio.ensure_future(self._rate_limited(
                sem, self._fetch_search(sub, query, limit=100, tier=1, pages=2, min_score=min_score)
            ))))

        for side_sources in sources.get("tier2_team_specific", {}).values():
            for src in side_sources:
                sub = src["subreddit"]
                min_score = self._src_min_score(src)
                primary_tasks.append((sub, asyncio.ensure_future(self._rate_limited(
                    sem, self._fetch_hot(sub, limit=50, tier=2, min_score=min_score)
                ))))
                if self._src_include_comments(src):
                    tier2_with_comments.append((sub, 2, min_score))

        for side, side_sources in sources.get("tier3_national_sentiment", {}).items():
            kw = f"{home_code} {away_code} world cup" if side == "home" else f"{away_code} {home_code} world cup"
            for src in side_sources:
                sub = src["subreddit"]
                min_score = self._src_min_score(src)
                primary_tasks.append((sub, asyncio.ensure_future(self._rate_limited(
                    sem, self._fetch_search(sub, kw, limit=30, tier=3, pages=1, min_score=min_score)
                ))))

        if not primary_tasks:
            return []

        results = await asyncio.gather(*(t for _, t in primary_tasks), return_exceptions=True)

        all_posts: list[FetchedPost] = []
        per_sub: dict[str, list[FetchedPost]] = defaultdict(list)
        for (sub, _), r in zip(primary_tasks, results):
            if isinstance(r, list):
                all_posts.extend(r)
                per_sub[sub].extend(r)
            elif isinstance(r, Exception):
                log.warning("reddit_task_failed", error=str(r))

        # IMPROVE-08: harvest comments from tier-2 hot threads.
        comment_tasks = []
        for sub, tier, min_score in tier2_with_comments:
            comment_tasks.append(self._rate_limited(
                sem,
                self._fetch_comments_for_top_posts(sub, tier, per_sub.get(sub, []), min_score=min_score),
            ))
        if comment_tasks:
            comment_results = await asyncio.gather(*comment_tasks, return_exceptions=True)
            for cr in comment_results:
                if isinstance(cr, list):
                    all_posts.extend(cr)

        log.info(
            "http_crawl_done",
            tasks=len(primary_tasks),
            comment_tasks=len(comment_tasks),
            posts=len(all_posts),
            **self.last_run_stats,
        )
        return _deduplicate(all_posts)
