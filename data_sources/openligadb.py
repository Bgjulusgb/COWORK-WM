"""OpenLigaDB connector — optional European cross-check (low weight).

Base : https://api.openligadb.de/   ·  Key: none  ·  Account: none  ·  Free
Scope: OpenLigaDB is club/league-centric (mostly German football) and has no
       reliable national-team result coverage, so the live path returns an empty
       list — the factors rely on openfootball for history. The connector exists
       so the architecture is complete and a richer query can drop in later; the
       mock supplies synthetic European data for tests.
Fallback: mock (European teams) / empty.
"""
from __future__ import annotations

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import openligadb_mock

log = structlog.get_logger("data_sources.openligadb")


class OpenLigaDBConnector(BaseConnector):
    connector_name = "openligadb"

    async def get_historical_results(self, code: str) -> FetchResult:
        code = code.upper()
        if settings.use_mock_openligadb:
            return FetchResult(openligadb_mock.historical_results(code), "mock", None, "mock")
        # No national-team coverage upstream → empty live result (not an error).
        return FetchResult([], "live", None, self.connector_name)


__all__ = ["OpenLigaDBConnector"]
