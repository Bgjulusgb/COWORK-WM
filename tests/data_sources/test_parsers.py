"""Live-path parser tests — the bits that break when an upstream schema drifts.

These exercise the pure parsing functions directly (no network), covering the
two openfootball JSON shapes, fixture/venue extraction, Open-Meteo hour-picking
and the RSS severity/alias model.
"""
from datetime import datetime, timezone

import pytest

from config.settings import settings
from data_sources.football_data_org import FootballDataOrgConnector, _parse_matches
from data_sources.openfootball import _parse_fixtures, _parse_worldcup_json, _venue_string
from data_sources.rss_news import _aliases, _entities, _get_nlp, _severity
from data_sources.weather import _nearest_hour_index, _parse


# ── openfootball history (two schemas) ───────────────────────────────────────
def test_openfootball_parses_top_level_matches_with_ft_score():
    raw = {"matches": [
        {"team1": "Brazil", "team2": "Mexico", "date": "2022-06-10", "score": {"ft": [2, 1]}},
    ]}
    out = _parse_worldcup_json(raw, 2022)
    assert len(out) == 1
    m = out[0]
    assert m.home_code == "BRA" and m.away_code == "MEX"
    assert m.home_score == 2 and m.away_score == 1 and m.is_finished


def test_openfootball_parses_rounds_nesting_with_score1_2():
    raw = {"rounds": [{"matches": [
        {"team1": {"name": "Argentina"}, "team2": {"name": "Brazil"},
         "date": "2018-06-20", "score1": 1, "score2": 1},
    ]}]}
    out = _parse_worldcup_json(raw, 2018)
    assert len(out) == 1
    assert out[0].home_code == "ARG" and out[0].away_code == "BRA"
    assert out[0].home_score == 1 and out[0].away_score == 1


def test_openfootball_skips_unmappable_team_names():
    raw = {"matches": [{"team1": "Atlantis", "team2": "El Dorado", "date": "2022-06-10"}]}
    assert _parse_worldcup_json(raw, 2022) == []


# ── fixtures + venue extraction ──────────────────────────────────────────────
def test_fixtures_capture_venue_and_unplayed_flag():
    raw = {"matches": [
        {"team1": "Mexico", "team2": "Canada", "date": "2026-06-13",
         "stadium": "Estadio Azteca", "city": "Mexico City"},
    ]}
    fixtures = _parse_fixtures(raw, 2026)
    assert len(fixtures) == 1
    assert fixtures[0].is_finished is False
    assert "Azteca" in fixtures[0].venue and "Mexico City" in fixtures[0].venue


def test_venue_string_handles_dict_stadium():
    assert _venue_string({"stadium": {"name": "SoFi Stadium"}, "city": "Los Angeles"}) == "SoFi Stadium, Los Angeles"
    assert _venue_string({}) is None


# ── Open-Meteo hour picking ──────────────────────────────────────────────────
def _hourly():
    return {"hourly": {
        "time": ["2026-06-13T17:00", "2026-06-13T18:00", "2026-06-13T19:00"],
        "temperature_2m": [20.0, 25.0, 30.0],
        "relative_humidity_2m": [50, 55, 60],
        "wind_speed_10m": [5, 6, 7],
        "precipitation": [0, 0, 1],
    }}


def test_weather_picks_nearest_hour():
    info = _parse(_hourly(), datetime(2026, 6, 13, 18, 10, tzinfo=timezone.utc))
    assert info is not None and info.temp_c == 25.0 and info.humidity_pct == 55


def test_weather_out_of_forecast_horizon_returns_none():
    assert _parse(_hourly(), datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)) is None


def test_nearest_hour_index_basic():
    times = ["2026-06-13T17:00", "2026-06-13T18:00"]
    assert _nearest_hour_index(times, datetime(2026, 6, 13, 17, 40, tzinfo=timezone.utc)) == 1


# ── RSS severity + alias model ───────────────────────────────────────────────
def test_rss_severity_orders_by_keyword():
    assert _severity("Star striker ruled out for the season") == 0.8
    assert _severity("Midfielder a doubt after a knock") == 0.4
    assert _severity("England win 3-0 in a friendly") == 0.0


def test_rss_aliases_include_nation_name():
    assert "england" in _aliases("ENG")
    # Short tokens are excluded to avoid false positives.
    assert all(len(a) >= 4 for a in _aliases("ENG"))


def test_rss_ner_resolves_country_to_code():
    nlp = _get_nlp()
    if nlp is None:
        pytest.skip("spaCy model not available")
    codes, persons = _entities(nlp, "England ruled out star midfielder ahead of the World Cup")
    assert "ENG" in codes        # GPE 'England' → FIFA code
    assert isinstance(persons, list)


# ── football-data.org ────────────────────────────────────────────────────────
def test_football_data_parse_matches():
    data = {"matches": [
        {"utcDate": "2026-06-13T18:00:00Z", "status": "FINISHED",
         "homeTeam": {"name": "Brazil", "tla": "BRA"},
         "awayTeam": {"name": "Mexico", "tla": "MEX"},
         "score": {"fullTime": {"home": 2, "away": 0}}, "venue": "MetLife Stadium"},
    ]}
    out = _parse_matches(data, 2026)
    assert len(out) == 1
    assert out[0].home_code == "BRA" and out[0].away_code == "MEX"
    assert out[0].is_finished and out[0].home_score == 2
    assert out[0].venue == "MetLife Stadium"


@pytest.mark.asyncio
async def test_football_data_disabled_without_key(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_football_data", True)
    monkeypatch.setattr(settings, "football_data_api_key", "")
    res = await FootballDataOrgConnector().get_fixtures(2026)
    assert res.mode == "mock" and res.data == []
