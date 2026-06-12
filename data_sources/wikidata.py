"""Wikidata connector — best-effort squad availability (optional).

Base : https://query.wikidata.org/sparql   ·  Key: none  ·  Account: none  ·  Free
Status: query.wikidata.org is NOT in the project's network allowlist, so this
        connector is MOCKED BY DEFAULT (settings.use_mock_wikidata=true) and the
        SquadAvailabilityFactor's weight defaults to 0. Flip both to opt in.

Reliable squad/availability data needs structured parsing that Wikidata does not
expose cleanly per-tournament, so the live path is a documented best-effort: it
returns available=False (the factor then re-normalises itself out) rather than
guessing. The mock provides populated data so tests can exercise the live-shaped
path. A real SPARQL query can drop into `_live_squad` later without touching the
factor.
"""
from __future__ import annotations

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import wikidata_mock
from data_sources.schemas import SquadInfo

log = structlog.get_logger("data_sources.wikidata")


class WikidataConnector(BaseConnector):
    connector_name = "wikidata"

    async def get_squad_info(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_wikidata:
            return FetchResult(wikidata_mock.squad_info(code), "mock", None, "mock")
        return await self._live_squad(code)

    async def _live_squad(self, code: str) -> FetchResult:
        # Best-effort placeholder: without a robust per-tournament SPARQL query
        # we cannot extract trustworthy squad data, so we report unavailable and
        # let the ensemble drop this factor. Documented limitation, not a bug.
        log.debug("wikidata_live_unavailable", code=code)
        return FetchResult(SquadInfo(source="wikidata", code=code, available=False),
                           "live", None, self.connector_name)


__all__ = ["WikidataConnector"]
