"""Tests for the v3 data-source layer: venue resolution, weather/RSS mocks and
the orchestrator end-to-end in fully-mocked (offline) mode."""
from datetime import datetime, timezone

import pytest

from config.settings import settings
from data_sources import venues
from data_sources.mock import rss_mock, weather_mock
from data_sources.orchestrator import DataSourceOrchestrator, _rest_days
from data_sources.rss_news import RssNewsConnector
from data_sources.schemas import HistoricalMatch
from data_sources.weather import WeatherConnector
from factors.base import FactorContext

_KICK = datetime(2026, 6, 20, 19, 0, tzinfo=timezone.utc)


def test_venue_resolution_matches_keywords():
    v = venues.resolve("Estadio Azteca, Mexico City")
    assert v is not None and v.altitude_m == 2240
    assert venues.resolve("MetLife Stadium, New York/New Jersey").country == "USA"
    assert venues.resolve("totally unknown ground") is None


def test_weather_mock_hotter_at_low_latitude():
    miami = weather_mock.weather_for(25.9, -80.2, 3)
    seattle = weather_mock.weather_for(47.6, -122.3, 5)
    assert miami.temp_c > seattle.temp_c


def test_rss_mock_is_deterministic():
    assert rss_mock.injury_news("ENG") == rss_mock.injury_news("ENG")


def test_rest_days_recent_history():
    matches = [HistoricalMatch(source="t", home_code="HHH", away_code="X", home_name="H",
                               away_name="X", kickoff_utc=_KICK.replace(day=14),
                               home_score=1, away_score=0)]
    assert _rest_days(matches, _KICK) == 6


def test_rest_days_none_when_only_old_history():
    old = [HistoricalMatch(source="t", home_code="HHH", away_code="X", home_name="H",
                           away_name="X", kickoff_utc=datetime(2018, 6, 1, tzinfo=timezone.utc),
                           home_score=1, away_score=0)]
    assert _rest_days(old, _KICK) is None


@pytest.mark.asyncio
async def test_weather_connector_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_weather", True)
    res = await WeatherConnector().get_weather(25.9, -80.2, 3, _KICK)
    assert res.mode == "mock"
    assert res.data.temp_c is not None


@pytest.mark.asyncio
async def test_rss_connector_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_rss", True)
    res = await RssNewsConnector().get_team_news("ENG", "England")
    assert res.mode == "mock"
    assert isinstance(res.data, list)


@pytest.mark.asyncio
async def test_orchestrator_populates_context_offline(monkeypatch):
    # Force every source to its deterministic mock so the test never touches
    # the network and the modes are predictable.
    for flag in ("use_mock_openfootball", "use_mock_thesportsdb", "use_mock_openligadb",
                 "use_mock_wikidata", "use_mock_weather", "use_mock_rss"):
        monkeypatch.setattr(settings, flag, True)

    ctx = FactorContext(
        match_id="m1",
        config={"teams": {"home": {"name": "Brazil", "code": "BRA"},
                          "away": {"name": "Mexico", "code": "MEX"}}},
        home_code="BRA", away_code="MEX",
        kickoff_utc=_KICK, venue="Estadio Azteca, Mexico City",
    )
    await DataSourceOrchestrator().populate(ctx)

    assert ctx.historical_matches_home, "expected mock history"
    assert ctx.venue_info is not None and ctx.venue_info.altitude_m == 2240
    assert ctx.weather is not None
    assert ctx.provenance["history_home"]["mode"] == "mock"
    assert ctx.provenance["weather"]["mode"] == "mock"
