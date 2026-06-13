"""the-odds-api.com — live bookmaker odds for the WC-2026 competition.

API   : https://the-odds-api.com/   ·   v4 REST   ·   key in ``?apiKey=``
Free  : 500 requests/month on the free tier; per-call usage in response header.
Auth  : single API key, no OAuth, no account beyond signup.

The connector fetches the full event list for ``soccer_fifa_world_cup`` once
per kickoff window (cached) and returns the **median** odds across the bookies
the user requested (EU region by default — more stable than picking a single
book). Markets exposed:

* ``h2h``     → 1X2 ``[home, draw, away]``
* ``totals``  → Over/Under 2.5 ``[over, under]``
* ``btts``    → Both teams to score ``[yes, no]``

Failure-soft: any HTTP error / parse error / "no event matches" returns the
deterministic mock so the pipeline edge table never disappears entirely.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from typing import Any

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import odds_api_mock

log = structlog.get_logger("data_sources.odds_api")

_BASE = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
# Cache odds for ~30 min — bookies move on news cycles, no point caching hours.
_ODDS_TTL_S = 30 * 60
# Window we look for the fixture in: ±36h around kickoff covers timezone slop
# and the period during which a pre-match line is meaningful.
_WINDOW_HOURS = 36


class OddsApiConnector(BaseConnector):
    connector_name = "odds_api"

    async def get_odds(
        self,
        home_code: str,
        away_code: str,
        kickoff_utc: datetime | None,
        *,
        home_name: str | None = None,
        away_name: str | None = None,
    ) -> FetchResult:
        """Return decimal odds for ``home`` vs ``away`` around kickoff.

        Falls back to the deterministic mock when:
        * ``USE_MOCK_ODDS_API`` is set or no API key configured
        * the upstream returns an error / no events
        * no event in the response matches the team pair within the window
        """
        if settings.use_mock_odds_api or not settings.odds_api_key:
            return FetchResult(
                odds_api_mock.odds_for(home_code, away_code),
                "mock", None, "mock",
            )

        params = {
            "apiKey": settings.odds_api_key,
            "regions": "eu",
            "markets": "h2h,totals,btts",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if kickoff_utc is not None:
            window_lo = kickoff_utc - timedelta(hours=_WINDOW_HOURS)
            window_hi = kickoff_utc + timedelta(hours=_WINDOW_HOURS)
            params["commenceTimeFrom"] = window_lo.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["commenceTimeTo"] = window_hi.strftime("%Y-%m-%dT%H:%M:%SZ")

        res = await self._get_json(_BASE, params=params, ttl_s=_ODDS_TTL_S)
        if not res.ok:
            return FetchResult(
                odds_api_mock.odds_for(home_code, away_code),
                "mock", None, "mock",
            )

        parsed = _parse_event_list(
            res.data,
            home_code=home_code, away_code=away_code,
            home_name=home_name, away_name=away_name,
            kickoff_utc=kickoff_utc,
        )
        if parsed is None:
            log.info("odds_api_no_match", home=home_code, away=away_code)
            return FetchResult(
                odds_api_mock.odds_for(home_code, away_code),
                "mock", None, "mock",
            )
        return res.replace_data(parsed)


# ── parsing helpers (pure, unit-testable) ─────────────────────────────────────
def _normalise(name: str | None) -> str:
    return (name or "").strip().lower()


def _matches_team(side_text: str | None, code: str, name: str | None) -> bool:
    s = _normalise(side_text)
    if not s:
        return False
    if s == code.lower():
        return True
    if name:
        n = _normalise(name)
        if n == s or (n and (n in s or s in n)):
            return True
        # Token overlap on a ≥4-char word — catches the genuine football naming
        # drift like "South Korea" ↔ "Korea Republic", "Ivory Coast" ↔ "Côte
        # d'Ivoire". Short tokens ("of", "the") don't qualify, so unrelated
        # teams like "Bayern" and "Bavaria" still don't accidentally match.
        s_tokens = {t for t in s.split() if len(t) >= 4}
        n_tokens = {t for t in n.split() if len(t) >= 4}
        if s_tokens & n_tokens:
            return True
    return False


def _commence_to_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pick_event(
    events: list[dict[str, Any]],
    *,
    home_code: str, away_code: str,
    home_name: str | None, away_name: str | None,
    kickoff_utc: datetime | None,
) -> dict[str, Any] | None:
    """Find the event whose teams + kickoff best match the requested fixture."""
    best: tuple[float, dict[str, Any]] | None = None
    for ev in events or []:
        ht = ev.get("home_team")
        at = ev.get("away_team")
        # Try both orientations — bookies sometimes flip home/away from FIFA.
        forward = (_matches_team(ht, home_code, home_name)
                   and _matches_team(at, away_code, away_name))
        reverse = (_matches_team(ht, away_code, away_name)
                   and _matches_team(at, home_code, home_name))
        if not (forward or reverse):
            continue
        # Pick the event with the smallest time delta to our kickoff.
        ev_dt = _commence_to_dt(ev.get("commence_time"))
        if ev_dt and kickoff_utc:
            gap = abs((ev_dt - kickoff_utc).total_seconds())
        else:
            gap = 0.0
        ev["_orientation_reversed"] = reverse and not forward
        if best is None or gap < best[0]:
            best = (gap, ev)
    return best[1] if best else None


def _median_h2h(
    event: dict[str, Any], *, home_code: str, away_code: str,
    home_name: str | None, away_name: str | None,
) -> list[float] | None:
    """Median of every bookmaker's h2h line → [FIFA-home, draw, FIFA-away].

    Each outcome carries the team name directly, so home/away orientation
    flips in ``event.home_team`` / ``event.away_team`` (handled by
    ``_pick_event``) don't matter here: prices are routed to the home/away
    bucket purely by the outcome's ``name`` field matching the FIFA team.
    """
    home_prices: list[float] = []
    draw_prices: list[float] = []
    away_prices: list[float] = []
    for bk in event.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != "h2h":
                continue
            for o in mk.get("outcomes") or []:
                name = o.get("name")
                price = o.get("price")
                if price is None:
                    continue
                if _matches_team(name, home_code, home_name):
                    home_prices.append(float(price))
                elif _matches_team(name, away_code, away_name):
                    away_prices.append(float(price))
                elif _normalise(name) == "draw":
                    draw_prices.append(float(price))
    if not (home_prices and draw_prices and away_prices):
        return None
    return [round(median(home_prices), 3),
            round(median(draw_prices), 3),
            round(median(away_prices), 3)]


def _median_totals(event: dict[str, Any], line: float = 2.5) -> list[float] | None:
    over_prices: list[float] = []
    under_prices: list[float] = []
    for bk in event.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != "totals":
                continue
            for o in mk.get("outcomes") or []:
                price = o.get("price")
                point = o.get("point")
                if price is None or point is None or abs(float(point) - line) > 1e-9:
                    continue
                if _normalise(o.get("name")) == "over":
                    over_prices.append(float(price))
                elif _normalise(o.get("name")) == "under":
                    under_prices.append(float(price))
    if not (over_prices and under_prices):
        return None
    return [round(median(over_prices), 3), round(median(under_prices), 3)]


def _median_btts(event: dict[str, Any]) -> list[float] | None:
    yes_prices: list[float] = []
    no_prices: list[float] = []
    for bk in event.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") not in ("btts", "both_teams_to_score"):
                continue
            for o in mk.get("outcomes") or []:
                price = o.get("price")
                if price is None:
                    continue
                name = _normalise(o.get("name"))
                if name == "yes":
                    yes_prices.append(float(price))
                elif name == "no":
                    no_prices.append(float(price))
    if not (yes_prices and no_prices):
        return None
    return [round(median(yes_prices), 3), round(median(no_prices), 3)]


def _parse_event_list(
    data: Any, *,
    home_code: str, away_code: str,
    home_name: str | None, away_name: str | None,
    kickoff_utc: datetime | None,
) -> dict[str, list[float]] | None:
    """Reduce the upstream event list to a single ``{market: [...]}`` dict."""
    if not isinstance(data, list):
        return None
    event = _pick_event(
        data,
        home_code=home_code, away_code=away_code,
        home_name=home_name, away_name=away_name,
        kickoff_utc=kickoff_utc,
    )
    if event is None:
        return None
    out: dict[str, list[float]] = {}
    h2h = _median_h2h(event, home_code=home_code, away_code=away_code,
                      home_name=home_name, away_name=away_name)
    if h2h is not None:
        out["1x2"] = h2h
    totals = _median_totals(event)
    if totals is not None:
        out["ou_2_5"] = totals
    btts = _median_btts(event)
    if btts is not None:
        out["btts"] = btts
    return out or None


__all__ = ["OddsApiConnector"]
