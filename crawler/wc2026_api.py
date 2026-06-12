from __future__ import annotations

import asyncio
import os
import ssl
from typing import Any

import httpx
import structlog

log = structlog.get_logger("crawler.wc2026_api")

_BASE = "https://worldcup26.ir/get"
_TIMEOUT = 10.0
_MAX_RETRIES = 3


def _ssl_context():
    """worldcup26.ir uses a certificate not trusted by standard CA bundles.
    Set WC_API_SKIP_SSL=1 to disable verification for this public data API."""
    if os.environ.get("WC_API_SKIP_SSL", "1") == "1":
        return False
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return True


# BUG-09 fix: cache one client + connection pool instead of building/closing
# a fresh one on every request. Closed on application shutdown.
_CLIENT: httpx.AsyncClient | None = None
_CLIENT_LOCK = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = httpx.AsyncClient(timeout=_TIMEOUT, verify=_ssl_context())
    return _CLIENT


async def close_client() -> None:
    """Lifespan hook — release the cached connection pool on shutdown."""
    global _CLIENT
    if _CLIENT is not None:
        try:
            await _CLIENT.aclose()
        finally:
            _CLIENT = None


async def _get(path: str) -> list[dict[str, Any]] | dict[str, Any] | None:
    url = f"{_BASE}/{path}"
    client = await _get_client()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            log.warning("wc_api_http_error", url=url, status=resp.status_code, attempt=attempt)
        except Exception as exc:
            log.warning("wc_api_fetch_failed", url=url, error=str(exc), attempt=attempt)
        if attempt < _MAX_RETRIES:
            # Exponential backoff: 2, 4 seconds.
            await asyncio.sleep(2 ** attempt)
    return None


async def fetch_games() -> list[dict[str, Any]]:
    data = await _get("games")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("games", data.get("data", []))
    return []


async def fetch_groups() -> list[dict[str, Any]]:
    data = await _get("groups")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("groups", data.get("data", []))
    return []


async def fetch_teams() -> list[dict[str, Any]]:
    data = await _get("teams")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("teams", data.get("data", []))
    return []


async def fetch_stadiums() -> list[dict[str, Any]]:
    data = await _get("stadiums")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("stadiums", data.get("data", []))
    return []
