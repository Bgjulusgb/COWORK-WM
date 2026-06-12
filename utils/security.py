"""IMPROVE-17: API-side rate limiting + admin auth.

Dependency-free implementation: a per-IP sliding-window counter held in
memory. Process-local — fine for the single-uvicorn-worker MVP. For a
fleet of workers swap to a Redis-backed limiter behind the same interface.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import Header, HTTPException, Request

from config.settings import settings


class RateLimiter:
    def __init__(self, requests: int, per_seconds: int = 60) -> None:
        self.requests = requests
        self.per_seconds = per_seconds
        self._hits: defaultdict[str, Deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> None:
        if self.requests <= 0:
            return
        now = time.monotonic()
        cutoff = now - self.per_seconds
        bucket = self._hits[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.requests:
            raise HTTPException(429, "Rate limit exceeded — bitte einen Moment warten.")
        bucket.append(now)


_get_limiter = RateLimiter(settings.rate_limit_get_per_minute)
_post_limiter = RateLimiter(settings.rate_limit_post_per_minute)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_get(request: Request) -> None:
    _get_limiter.hit(_client_ip(request))


def rate_limit_post(request: Request) -> None:
    _post_limiter.hit(_client_ip(request))


def require_admin_key(x_api_key: str = Header(default="")) -> None:
    """Block when ADMIN_API_KEY is set and the header doesn't match.

    A blank ADMIN_API_KEY (default) leaves the endpoint open so a fresh
    install still works without configuration.
    """
    expected = settings.admin_api_key or ""
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(401, "Missing or invalid X-API-Key")
