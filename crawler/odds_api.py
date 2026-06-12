"""EXTEND-02: bookmaker odds via the-odds-api.com.

Free tier: 500 requests/month. We query the FIFA World Cup market once per
crawl and cache the result for `cache_ttl_s` so a flurry of refreshes shares
one upstream call.

Settings:
    ODDS_API_KEY=...            # required, otherwise the integration is disabled
    ENABLE_ODDS_INTEGRATION=true
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
import structlog

from config.settings import settings

log = structlog.get_logger("crawler.odds_api")

_BASE = "https://api.the-odds-api.com/v4"
_SPORT_KEY = "soccer_fifa_world_cup"
_REGIONS = "eu,uk,us"
_MARKETS = "h2h"  # 1X2 implied probability
_CACHE_TTL_S = 600  # 10 minutes
_TIMEOUT = 8.0


_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CLIENT: httpx.AsyncClient | None = None


async def _client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(timeout=_TIMEOUT)
    return _CLIENT


async def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        try:
            await _CLIENT.aclose()
        finally:
            _CLIENT = None


def _is_enabled() -> bool:
    return bool(settings.enable_odds_integration and settings.odds_api_key)


async def fetch_odds() -> list[dict]:
    """Fetch all upcoming WC fixtures with H2H odds. Cached for _CACHE_TTL_S.

    Returns an empty list when the integration is disabled or the API errors.
    """
    if not _is_enabled():
        return []

    cached = _CACHE.get(_SPORT_KEY)
    if cached and time.time() - cached[0] < _CACHE_TTL_S:
        return cached[1]

    url = f"{_BASE}/sports/{_SPORT_KEY}/odds"
    params = {
        "regions": _REGIONS,
        "markets": _MARKETS,
        "oddsFormat": "decimal",
        "apiKey": settings.odds_api_key,
    }
    try:
        resp = await (await _client()).get(url, params=params)
    except Exception as exc:
        log.warning("odds_api_fetch_failed", error=str(exc))
        return []
    if resp.status_code != 200:
        log.warning("odds_api_status", status=resp.status_code)
        return []
    data = resp.json()
    if not isinstance(data, list):
        return []
    _CACHE[_SPORT_KEY] = (time.time(), data)
    return data


def _decimal_to_implied(odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def implied_probabilities(home_name: str, away_name: str, fixtures: list[dict]) -> Optional[dict]:
    """Best-effort lookup. Strips overround so the three probs sum to 1."""
    home_l, away_l = home_name.lower(), away_name.lower()
    for fixture in fixtures:
        ht = (fixture.get("home_team") or "").lower()
        at = (fixture.get("away_team") or "").lower()
        if home_l not in ht and home_l not in at:
            continue
        if away_l not in ht and away_l not in at:
            continue

        bookmakers = fixture.get("bookmakers") or []
        # Average across bookmakers for stability.
        sums = {"home": 0.0, "draw": 0.0, "away": 0.0}
        counts = 0
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                triple: dict[str, float] = {}
                for o in outcomes:
                    name = (o.get("name") or "").lower()
                    price = float(o.get("price", 0) or 0)
                    if not price:
                        continue
                    if "draw" in name or name == "tie":
                        triple["draw"] = _decimal_to_implied(price)
                    elif name in (ht, fixture.get("home_team", "").lower()):
                        triple["home"] = _decimal_to_implied(price)
                    elif name in (at, fixture.get("away_team", "").lower()):
                        triple["away"] = _decimal_to_implied(price)
                if len(triple) == 3:
                    for k, v in triple.items():
                        sums[k] += v
                    counts += 1
        if counts == 0:
            return None
        avg = {k: v / counts for k, v in sums.items()}
        total = sum(avg.values())
        if total <= 0:
            return None
        # Normalise to strip the bookmaker overround / vig.
        return {k: v / total for k, v in avg.items()}
    return None
