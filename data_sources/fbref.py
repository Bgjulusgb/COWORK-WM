"""FBref HTML xG connector — pandas.read_html + selectolax fast-parse.

Real FBref pages are JS-rendered tables but the underlying HTML is shipped in
the document, so a `pandas.read_html()` over the cached HTML pulls out the
team-match-log table without a browser. We pre-parse with selectolax to drop
the navigation chunks (≈8x faster than BeautifulSoup) and then hand the table
slice to pandas.

International squads don't have club URLs; the v3 fallback is to read the
national-team page (one URL per FIFA code, looked up against a small table).
When the page changes shape (FBref re-skins ~once a year), the connector
degrades to the deterministic mock and the orchestrator records mode="error".
"""
from __future__ import annotations

import asyncio
from io import StringIO
from typing import Optional

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import fbref_mock
from data_sources.schemas import XgInfo

log = structlog.get_logger("data_sources.fbref")

# FIFA 3-letter code → FBref nation-team page slug. Only confederations with
# real national-team coverage are listed; others fall back to mock.
_FBREF_SLUG = {
    "GER": "Germany-Men", "FRA": "France-Men", "ENG": "England-Men",
    "ESP": "Spain-Men", "ITA": "Italy-Men", "POR": "Portugal-Men",
    "NED": "Netherlands-Men", "BEL": "Belgium-Men", "CRO": "Croatia-Men",
    "BRA": "Brazil-Men", "ARG": "Argentina-Men", "URU": "Uruguay-Men",
    "COL": "Colombia-Men", "MEX": "Mexico-Men", "USA": "United-States-Men",
    "CAN": "Canada-Men", "JPN": "Japan-Men", "KOR": "South-Korea-Men",
    "AUS": "Australia-Men", "MAR": "Morocco-Men", "SEN": "Senegal-Men",
    "EGY": "Egypt-Men", "NGA": "Nigeria-Men", "GHA": "Ghana-Men",
    "TUN": "Tunisia-Men", "IRN": "Iran-Men", "KSA": "Saudi-Arabia-Men",
    "QAT": "Qatar-Men", "SUI": "Switzerland-Men", "POL": "Poland-Men",
    "DEN": "Denmark-Men", "AUT": "Austria-Men", "SRB": "Serbia-Men",
    "ECU": "Ecuador-Men", "CHI": "Chile-Men", "PER": "Peru-Men",
    "PAR": "Paraguay-Men", "WAL": "Wales-Men", "SCO": "Scotland-Men",
    "TUR": "Turkey-Men", "CZE": "Czech-Republic-Men", "SVK": "Slovakia-Men",
}
_FBREF_BASE = "https://fbref.com/en/squads"
_FBREF_TTL_S = 6 * 3600.0  # xG numbers update only when the team plays


class FbrefConnector(BaseConnector):
    connector_name = "fbref"

    async def get_team_xg(self, code: str, last_n: int = 10) -> FetchResult:
        code = code.upper()
        if settings.use_mock_fbref:
            return FetchResult(fbref_mock.team_xg(code, last_n), "mock", None, "mock")

        slug = _FBREF_SLUG.get(code)
        if slug is None:
            log.debug("fbref_no_slug", code=code)
            return FetchResult(fbref_mock.team_xg(code, last_n), "mock", None, "mock")

        url = f"{_FBREF_BASE}/{slug}"
        res = await self._get_text(url, ttl_s=_FBREF_TTL_S)
        if not res.ok or not isinstance(res.data, str):
            return FetchResult(fbref_mock.team_xg(code, last_n), "mock", None, "mock")

        parsed = await asyncio.to_thread(self._parse_team_xg, res.data, code, last_n)
        if parsed is None:
            return FetchResult(fbref_mock.team_xg(code, last_n), "mock", None, "mock")
        return res.replace_data(parsed)

    def _parse_team_xg(self, html: str, code: str, last_n: int) -> Optional[XgInfo]:
        try:
            import pandas as pd
            from selectolax.parser import HTMLParser
        except Exception:
            return None
        try:
            tree = HTMLParser(html)
            # FBref drops every match-log into a <table id="matchlogs_for"> with
            # xG / xGA / SoT columns. Slice that one HTML fragment for pandas.
            node = tree.css_first("table#matchlogs_for") or tree.css_first("table#matchlogs_all")
            if node is None:
                return None
            tables = pd.read_html(StringIO(node.html), header=0)
            if not tables:
                return None
            df = tables[0].dropna(how="all").tail(last_n)
        except Exception as exc:
            log.debug("fbref_parse_failed", code=code, error=str(exc))
            return None

        def _col(*names: str):
            for n in names:
                if n in df.columns:
                    return df[n]
            return None

        xg_for = _col("xG", "xG_for")
        xg_ag = _col("xGA", "xG_ag")
        sot = _col("SoT", "Shots on Target")
        gf = _col("GF", "Goals For")
        ga = _col("GA", "Goals Against")

        def _avg(s):
            if s is None:
                return None
            try:
                return float(s.astype(float).mean())
            except Exception:
                return None

        return XgInfo(
            source="fbref",
            code=code,
            matches_considered=int(len(df)),
            xg_for_avg=_avg(xg_for),
            xg_against_avg=_avg(xg_ag),
            shots_on_target_avg=_avg(sot),
            goals_for_avg=_avg(gf),
            goals_against_avg=_avg(ga),
        )


__all__ = ["FbrefConnector"]
