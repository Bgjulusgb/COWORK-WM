"""Transfermarkt HTML connector — squad market values.

Real Transfermarkt aggressively throttles bots; we keep concurrency=2 and
respect ``transfermarkt_request_gap_s`` (default 1.5s) between calls. Output
schema is :class:`SquadValueInfo`; the SquadValueFactor takes the log ratio
of the two sides' totals as a tilt signal.

Parsing strategy: selectolax (≈8x faster than BeautifulSoup) finds the
``table.items > tbody > tr`` rows; ``span.rechts.hauptlink`` per row carries
the player market value. We sum and report the squad totals. When the markup
shifts (Transfermarkt re-skins ~yearly), the connector returns mock and the
SquadValueFactor self-disables.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import transfermarkt_mock
from data_sources.schemas import SquadValueInfo

log = structlog.get_logger("data_sources.transfermarkt")

# FIFA → Transfermarkt national-team slug + id (national-team URLs are stable).
_TM_NATION = {
    "GER": ("deutschland", 3262),  "FRA": ("frankreich", 3377),
    "ENG": ("england", 3299),      "ESP": ("spanien", 3375),
    "ITA": ("italien", 3376),      "POR": ("portugal", 3300),
    "NED": ("niederlande", 3382),  "BEL": ("belgien", 3382),
    "CRO": ("kroatien", 3375),     "BRA": ("brasilien", 26),
    "ARG": ("argentinien", 9),     "URU": ("uruguay", 109),
    "COL": ("kolumbien", 25),      "MEX": ("mexiko", 102),
    "USA": ("vereinigte-staaten-von-amerika", 207),
    "CAN": ("kanada", 49),         "JPN": ("japan", 53),
    "KOR": ("korea-sued", 100),    "AUS": ("australien", 32),
    "MAR": ("marokko", 91),        "SEN": ("senegal", 168),
    "EGY": ("aegypten", 41),       "NGA": ("nigeria", 110),
    "GHA": ("ghana", 51),          "TUN": ("tunesien", 113),
    "IRN": ("iran", 89),           "KSA": ("saudi-arabien", 167),
    "QAT": ("katar", 128),         "SUI": ("schweiz", 148),
    "POL": ("polen", 125),         "DEN": ("daenemark", 81),
}
_TM_BASE = "https://www.transfermarkt.com/{slug}/startseite/verein/{tid}"
_TM_TTL_S = 24 * 3600.0  # market values change daily, weekly is enough

_VALUE_RE = re.compile(r"€\s*([\d.,]+)\s*([kmM]?)", re.IGNORECASE)

# Bound concurrency across all instances so a multi-match crawl can't hammer.
_GLOBAL_SEM = asyncio.Semaphore(2)
_LAST_FETCH_TS = 0.0
_LAST_FETCH_LOCK = asyncio.Lock()


def _parse_value(raw: str) -> Optional[float]:
    m = _VALUE_RE.search(raw)
    if not m:
        return None
    num = m.group(1).replace(".", "").replace(",", ".")
    try:
        value = float(num)
    except ValueError:
        return None
    suffix = m.group(2).lower()
    if suffix == "m":
        return value * 1_000_000
    if suffix == "k":
        return value * 1_000
    return value


class TransfermarktConnector(BaseConnector):
    connector_name = "transfermarkt"

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def _polite_fetch(self, url: str) -> FetchResult:
        global _LAST_FETCH_TS
        async with _GLOBAL_SEM:
            async with _LAST_FETCH_LOCK:
                now = asyncio.get_event_loop().time()
                gap = settings.transfermarkt_request_gap_s
                if now - _LAST_FETCH_TS < gap:
                    await asyncio.sleep(gap - (now - _LAST_FETCH_TS))
                _LAST_FETCH_TS = asyncio.get_event_loop().time()
            return await self._get_text(url, ttl_s=_TM_TTL_S)

    async def get_squad_value(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_transfermarkt:
            return FetchResult(transfermarkt_mock.squad_value(code), "mock", None, "mock")

        nation = _TM_NATION.get(code)
        if nation is None:
            return FetchResult(transfermarkt_mock.squad_value(code), "mock", None, "mock")

        slug, tid = nation
        url = _TM_BASE.format(slug=slug, tid=tid)
        res = await self._polite_fetch(url)
        if not res.ok or not isinstance(res.data, str):
            return FetchResult(transfermarkt_mock.squad_value(code), "mock", None, "mock")

        parsed = await asyncio.to_thread(self._parse_squad_value, res.data, code)
        if parsed is None:
            return FetchResult(transfermarkt_mock.squad_value(code), "mock", None, "mock")
        return res.replace_data(parsed)

    def _parse_squad_value(self, html: str, code: str) -> Optional[SquadValueInfo]:
        try:
            from selectolax.parser import HTMLParser
        except Exception:
            return None
        try:
            tree = HTMLParser(html)
            rows = tree.css("table.items > tbody > tr")
            values: list[float] = []
            for row in rows:
                cells = row.css("span.rechts.hauptlink, td.rechts.hauptlink")
                for cell in cells:
                    val = _parse_value(cell.text(strip=True) or "")
                    if val is not None and val > 0:
                        values.append(val)
                        break
        except Exception as exc:
            log.debug("transfermarkt_parse_failed", code=code, error=str(exc))
            return None
        if not values:
            return None
        squad_size = len(values)
        total = sum(values)
        top11 = sum(sorted(values, reverse=True)[:11])
        return SquadValueInfo(
            source="transfermarkt",
            code=code,
            total_value_eur=total,
            squad_size=squad_size,
            avg_value_eur=total / squad_size,
            top11_value_eur=top11,
        )


__all__ = ["TransfermarktConnector"]
