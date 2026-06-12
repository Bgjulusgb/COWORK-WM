"""Parallel Reddit crawler — runs HttpRedditCrawler + ArcticShiftCrawler simultaneously.

Both sources run via asyncio.gather. Results are merged and deduplicated using the
original Reddit base36 post ID so the same post from both sources counts only once.
Reddit JSON data takes priority over Arctic Shift when a duplicate is found.

Enable via USE_ARCTIC_SHIFT=true in .env.
"""
from __future__ import annotations

import asyncio

import structlog

from crawler.arctic_shift import ArcticShiftCrawler
from crawler.http_reddit import HttpRedditCrawler
from crawler.mock_reddit import FetchedPost

log = structlog.get_logger("crawler.parallel_reddit")


def _cross_source_deduplicate(posts: list[FetchedPost]) -> list[FetchedPost]:
    """Deduplicate across sources. Prefers source_post_id (original Reddit base36 ID)."""
    seen: set[str] = set()
    unique: list[FetchedPost] = []
    for p in posts:
        key = p.source_post_id or f"{p.subreddit}:{p.post_id}"
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


class ParallelRedditCrawler:
    """Runs HttpRedditCrawler and ArcticShiftCrawler in parallel.

    Drop-in replacement for HttpRedditCrawler — same crawl(config) interface.
    """

    def __init__(self) -> None:
        self._http = HttpRedditCrawler()
        self._arctic = ArcticShiftCrawler()

    async def crawl(self, config: dict) -> list[FetchedPost]:
        http_result, arctic_result = await asyncio.gather(
            self._http.crawl(config),
            self._arctic.crawl_for_match(config),
            return_exceptions=True,
        )

        http_posts: list[FetchedPost] = http_result if isinstance(http_result, list) else []
        arctic_posts: list[FetchedPost] = arctic_result if isinstance(arctic_result, list) else []

        if isinstance(http_result, Exception):
            log.warning("parallel_http_failed", error=str(http_result))
        if isinstance(arctic_result, Exception):
            log.warning("parallel_arctic_failed", error=str(arctic_result))

        # Reddit JSON first — wins on duplicates
        merged = _cross_source_deduplicate(http_posts + arctic_posts)
        log.info(
            "parallel_crawl_done",
            http=len(http_posts),
            arctic=len(arctic_posts),
            merged=len(merged),
        )
        return merged

    async def aclose(self) -> None:
        await asyncio.gather(
            self._http.aclose(),
            self._arctic.aclose(),
            return_exceptions=True,
        )
