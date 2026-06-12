"""IMPROVE-15: in-memory TTL cache.

Lightweight enough that we don't pull in Redis for a single-process MVP, but
the interface (`get_or_set`) maps 1:1 to an aioredis client so we can swap it
later. Thread-safe; bounded by `max_entries` with LRU eviction.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable


class TTLCache:
    def __init__(self, max_entries: int = 1024) -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_entries
        self._lock = asyncio.Lock()

    def _evict_locked(self) -> None:
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            # LRU touch.
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl_s: float) -> None:
        async with self._lock:
            self._data[key] = (time.monotonic() + ttl_s, value)
            self._data.move_to_end(key)
            self._evict_locked()

    async def invalidate(self, prefix: str | None = None) -> int:
        async with self._lock:
            if prefix is None:
                count = len(self._data)
                self._data.clear()
                return count
            keys = [k for k in self._data if k.startswith(prefix)]
            for k in keys:
                self._data.pop(k, None)
            return len(keys)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        ttl_s: float,
    ) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await factory()
        await self.set(key, value, ttl_s)
        return value


cache = TTLCache()
