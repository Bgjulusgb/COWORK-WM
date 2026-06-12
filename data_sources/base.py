"""Shared base for every external data-source connector.

Design mirrors crawler/wc2026_api.py (cached httpx client + exponential
backoff) but adds a two-stage TTL cache so the factors can run repeatedly
without hammering upstream:

    1. in-memory  utils.cache.cache         (fast path, per process)
    2. DataSourceCache table                (slow path, survives restarts)

Every public connector method returns a FetchResult so the caller knows the
provenance (live / cache / mock) — that powers the DataSourceBadge in the UI.
All failures degrade to FetchResult(mode="error", data=None); a connector
never raises into the factor layer.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

import httpx
import structlog

from config.settings import settings
from utils.cache import cache

log = structlog.get_logger("data_sources.base")


@dataclass
class FetchResult:
    """A connector response plus where it came from.

    `data` holds either raw JSON (from _get_json) or, after a connector parses
    it, the domain DTOs. `mode` is one of live | cache | mock | error.
    """
    data: Any
    mode: str
    fetched_at: datetime | None = None
    source: str = ""

    @property
    def ok(self) -> bool:
        return self.data is not None and self.mode != "error"

    def replace_data(self, data: Any) -> "FetchResult":
        """Return a copy with parsed data but the same provenance."""
        return FetchResult(data=data, mode=self.mode, fetched_at=self.fetched_at, source=self.source)


class BaseConnector:
    """HTTP + caching plumbing shared by all live connectors.

    Subclasses set `connector_name` (used as cache namespace + client key) and
    call `self._get_json(url, params, ttl_s=...)`.
    """
    connector_name: ClassVar[str] = "base"

    # One shared client pool per connector_name, closed on app shutdown.
    _clients: ClassVar[dict[str, httpx.AsyncClient]] = {}
    _client_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def _default_headers(self) -> dict[str, str]:
        """Headers for this connector's client. Subclasses override to add auth
        (e.g. football-data.org's X-Auth-Token)."""
        return {"User-Agent": settings.reddit_user_agent}

    async def _get_client(self) -> httpx.AsyncClient:
        client = BaseConnector._clients.get(self.connector_name)
        if client is not None:
            return client
        async with BaseConnector._client_lock:
            client = BaseConnector._clients.get(self.connector_name)
            if client is None:
                client = httpx.AsyncClient(
                    timeout=settings.datasource_http_timeout_s,
                    headers=self._default_headers(),
                    follow_redirects=True,
                )
                BaseConnector._clients[self.connector_name] = client
        return client

    @classmethod
    async def close_all(cls) -> None:
        """Lifespan hook — release every cached connection pool."""
        for name, client in list(cls._clients.items()):
            try:
                await client.aclose()
            finally:
                cls._clients.pop(name, None)

    def _cache_key(self, url: str, params: dict | None) -> str:
        raw = f"{self.connector_name}|{url}|{json.dumps(params or {}, sort_keys=True)}"
        return "ds:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    async def _get_json(
        self,
        url: str,
        params: dict | None = None,
        *,
        ttl_s: float | None = None,
    ) -> FetchResult:
        """GET url, JSON-decoded, through the two-stage cache + retry layer."""
        ttl = ttl_s if ttl_s is not None else settings.datasource_cache_ttl_hours * 3600.0
        key = self._cache_key(url, params)

        mem = await cache.get(key)
        if mem is not None:
            data, fetched_at = mem
            return FetchResult(data, "cache", fetched_at, self.connector_name)

        db = await self._db_cache_get(key)
        if db is not None:
            data, fetched_at = db
            await cache.set(key, (data, fetched_at), ttl)
            return FetchResult(data, "cache", fetched_at, self.connector_name)

        data = await self._fetch_with_retry(url, params)
        if data is None:
            return FetchResult(None, "error", None, self.connector_name)

        fetched_at = datetime.now(timezone.utc)
        await cache.set(key, (data, fetched_at), ttl)
        await self._db_cache_set(key, url, data, ttl)
        return FetchResult(data, "live", fetched_at, self.connector_name)

    async def _get_text(
        self,
        url: str,
        params: dict | None = None,
        *,
        ttl_s: float | None = None,
    ) -> FetchResult:
        """Like _get_json but returns the raw response text (RSS/XML feeds)."""
        ttl = ttl_s if ttl_s is not None else settings.datasource_cache_ttl_hours * 3600.0
        key = self._cache_key(url, params) + ":text"

        mem = await cache.get(key)
        if mem is not None:
            data, fetched_at = mem
            return FetchResult(data, "cache", fetched_at, self.connector_name)

        db = await self._db_cache_get(key)
        if db is not None:
            data, fetched_at = db
            await cache.set(key, (data, fetched_at), ttl)
            return FetchResult(data, "cache", fetched_at, self.connector_name)

        text = await self._fetch_with_retry(url, params, as_text=True)
        if text is None:
            return FetchResult(None, "error", None, self.connector_name)

        fetched_at = datetime.now(timezone.utc)
        await cache.set(key, (text, fetched_at), ttl)
        await self._db_cache_set(key, url, text, ttl)
        return FetchResult(text, "live", fetched_at, self.connector_name)

    async def _fetch_with_retry(self, url: str, params: dict | None, *, as_text: bool = False) -> Any | None:
        # Contract-Härtung (Phase 6.1): AUCH Fehler bei der Client-Konstruktion
        # (z.B. SOCKS-Proxy ohne socksio, kaputte Env-Proxies) dürfen nie aus dem
        # Konnektor herauspropagieren — sonst endet der Slice als `error` ohne
        # Mock-Fallback statt sauber zu degradieren (CLAUDE.md-Contract).
        try:
            client = await self._get_client()
        except Exception as exc:
            log.warning("ds_client_init_failed",
                        connector=self.connector_name, error=str(exc))
            return None
        attempts = max(1, settings.datasource_retry_attempts)
        backoff = settings.datasource_retry_backoff_s
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    return resp.text if as_text else resp.json()
                log.warning(
                    "ds_http_status",
                    connector=self.connector_name, url=url,
                    status=resp.status_code, attempt=attempt,
                )
                # Permanent client errors (not 429) won't improve on retry.
                if resp.status_code != 429 and 400 <= resp.status_code < 500:
                    return None
            except Exception as exc:
                log.warning(
                    "ds_fetch_failed",
                    connector=self.connector_name, url=url,
                    error=str(exc), attempt=attempt,
                )
            if attempt < attempts:
                await asyncio.sleep(backoff * attempt)
        return None

    # ── persistent cache backstop (best-effort; SQLite lock errors swallowed) ──
    async def _db_cache_get(self, key: str) -> tuple[Any, datetime | None] | None:
        try:
            from db.database import AsyncSessionLocal
            from db.models import DataSourceCache
            async with AsyncSessionLocal() as s:
                row = await s.get(DataSourceCache, key)
                if row is None:
                    return None
                if row.expires_at and row.expires_at < datetime.utcnow():
                    return None
                fetched = (
                    row.fetched_at.replace(tzinfo=timezone.utc) if row.fetched_at else None
                )
                return row.payload, fetched
        except Exception as exc:
            log.debug("ds_db_cache_get_failed", connector=self.connector_name, error=str(exc))
            return None

    async def _db_cache_set(self, key: str, endpoint: str, payload: Any, ttl: float) -> None:
        # Use an atomic upsert so concurrent coroutines sharing the same
        # cache_key (e.g. 32 parallel team lookups hitting the same WC JSON)
        # don't race: the SELECT-then-INSERT pattern has a TOCTOU gap that
        # causes UNIQUE constraint failures when multiple writers commit before
        # any of them sees the row the others inserted.
        try:
            from db.database import AsyncSessionLocal
            from db.models import DataSourceCache
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            async with AsyncSessionLocal() as s:
                now = datetime.utcnow()
                expires = now + timedelta(seconds=ttl)
                stmt = (
                    sqlite_insert(DataSourceCache)
                    .values(
                        cache_key=key,
                        connector=self.connector_name,
                        endpoint=endpoint,
                        payload=payload,
                        fetched_at=now,
                        expires_at=expires,
                    )
                    .on_conflict_do_update(
                        index_elements=["cache_key"],
                        set_={
                            "payload": payload,
                            "fetched_at": now,
                            "expires_at": expires,
                        },
                    )
                )
                await s.execute(stmt)
                await s.commit()
        except Exception as exc:
            log.debug("ds_db_cache_set_failed", connector=self.connector_name, error=str(exc))


__all__ = ["BaseConnector", "FetchResult"]
