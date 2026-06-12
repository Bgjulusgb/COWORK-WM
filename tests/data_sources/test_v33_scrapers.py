"""v3.3 scraper smoke-tests: exercise mock fall-back paths so the orchestrator
can rely on each connector returning a well-typed payload regardless of
whether the live endpoint is reachable.
"""
import pytest

from data_sources.fbref import FbrefConnector
from data_sources.fotmob import FotMobConnector
from data_sources.nvidia_llm import NvidiaLlmConnector
from data_sources.schemas import LineupInfo, SquadValueInfo, XgInfo
from data_sources.sofascore import SofaScoreConnector
from data_sources.transfermarkt import TransfermarktConnector
from data_sources.understat import UnderstatConnector


@pytest.mark.asyncio
async def test_fbref_mock_returns_xg_info(monkeypatch):
    monkeypatch.setattr("config.settings.settings.use_mock_fbref", True)
    res = await FbrefConnector().get_team_xg("GER", last_n=8)
    assert res.mode == "mock"
    assert isinstance(res.data, XgInfo)
    assert res.data.matches_considered == 8


@pytest.mark.asyncio
async def test_understat_mock_returns_xg_info(monkeypatch):
    monkeypatch.setattr("config.settings.settings.use_mock_understat", True)
    res = await UnderstatConnector().get_team_xg("ENG", last_n=10)
    assert res.mode == "mock"
    assert isinstance(res.data, XgInfo)
    assert res.data.xg_for_avg is not None


@pytest.mark.asyncio
async def test_fotmob_mock_lineup_and_injuries(monkeypatch):
    monkeypatch.setattr("config.settings.settings.use_mock_fotmob", True)
    c = FotMobConnector()
    lineup = await c.get_lineup("BRA")
    assert lineup.mode == "mock"
    assert isinstance(lineup.data, LineupInfo)
    inj = await c.get_injuries("BRA")
    assert inj.mode == "mock"
    assert isinstance(inj.data, list)


@pytest.mark.asyncio
async def test_sofascore_mock(monkeypatch):
    monkeypatch.setattr("config.settings.settings.use_mock_sofascore", True)
    c = SofaScoreConnector()
    lineup = await c.get_lineup("ARG")
    assert lineup.mode == "mock"
    inj = await c.get_injuries("ARG")
    assert inj.mode == "mock"


@pytest.mark.asyncio
async def test_transfermarkt_mock(monkeypatch):
    monkeypatch.setattr("config.settings.settings.use_mock_transfermarkt", True)
    res = await TransfermarktConnector().get_squad_value("FRA")
    assert res.mode == "mock"
    assert isinstance(res.data, SquadValueInfo)
    assert res.data.total_value_eur and res.data.total_value_eur > 0


@pytest.mark.asyncio
async def test_nvidia_llm_mock_when_disabled(monkeypatch):
    monkeypatch.setattr("config.settings.settings.use_nvidia_llm", False)
    monkeypatch.setattr("config.settings.settings.nvidia_api_key", "")
    res = await NvidiaLlmConnector().score_sentiment(
        ["test text 1", "test text 2"], "HOM", "AWA"
    )
    assert res.mode == "mock"
    assert "home" in res.data and "away" in res.data
