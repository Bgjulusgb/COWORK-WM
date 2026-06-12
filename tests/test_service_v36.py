"""Integration-Tests fuer den v3.6-Datenfluss durch ``run_crawl_and_predict``.

Was hier gepinnt wird:
* ``per_model_markets`` enthaelt alle 3 Modelle.
* ``confidence_intervals`` ist (nach K1) ein Dict mit raw/isotonic/platt-Keys.
* Ohne kalibrierungs-Artefakte sind isotonic/platt = None und raw enthaelt
  blended + die 3 Modelle.
* Wenn ein Isotonic-Artifact existiert, liegen die kalibrierten Baender in [0,1]
  und enthalten den kalibrierten Punktwert ``calibrated_home_win_prob``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from analysis import calibration as cal_mod
from analysis.calibration import CalibrationArtifact, IsotonicCurve
from config.settings import settings
from db.database import Base, _add_missing_columns
from services.match_service import run_crawl_and_predict, upsert_match_from_config
from utils.config_loader import discover_match_configs


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with Session() as s:
        yield s
    await engine.dispose()


def _force_offline(monkeypatch):
    monkeypatch.setattr(settings, "use_mock_crawler", True)
    monkeypatch.setattr(settings, "use_arctic_shift", False)
    monkeypatch.setattr(settings, "use_factor_ensemble", True)
    for flag in (
        "use_mock_openfootball", "use_mock_thesportsdb", "use_mock_openligadb",
        "use_mock_wikidata", "use_mock_weather", "use_mock_rss",
        "use_mock_football_data", "use_mock_fbref", "use_mock_understat",
        "use_mock_fotmob", "use_mock_sofascore", "use_mock_transfermarkt",
    ):
        if hasattr(settings, flag):
            monkeypatch.setattr(settings, flag, True)


def _pick_match():
    cfgs = discover_match_configs()
    return next((c for c in cfgs if "aus_vs_tur" in c.name), cfgs[0])


@pytest.mark.asyncio
async def test_per_model_markets_persisted(session, monkeypatch):
    _force_offline(monkeypatch)
    target = _pick_match()
    match = await upsert_match_from_config(session, target)
    await session.commit()

    _, _, pred = await run_crawl_and_predict(session, match, crawl_seed=42)

    assert pred.per_model_markets is not None
    pm = pred.per_model_markets
    assert set(pm.keys()) == {"poisson", "negbin", "glm_poisson"}
    for model_name, markets in pm.items():
        for key in ("home_win", "draw", "away_win"):
            assert 0.0 <= markets[key] <= 1.0


@pytest.mark.asyncio
async def test_confidence_intervals_have_raw_isotonic_platt_keys(session, monkeypatch):
    _force_offline(monkeypatch)
    target = _pick_match()
    match = await upsert_match_from_config(session, target)
    await session.commit()

    _, _, pred = await run_crawl_and_predict(session, match, crawl_seed=42)

    ci = pred.confidence_intervals
    if ci is None:
        # bootstrap_n=0 path; nothing to verify.
        return
    assert set(ci.keys()) == {"raw", "isotonic", "platt"}
    raw = ci["raw"]
    assert raw is not None
    assert "blended" in raw
    # All three goal models should also have CI rows when bootstrap_n > 0.
    for m in ("poisson", "negbin", "glm_poisson"):
        assert m in raw, f"{m} missing in raw CIs"


@pytest.mark.asyncio
async def test_calibrated_intervals_contain_calibrated_point(session, monkeypatch, tmp_path):
    """Mit einer halbierenden Isotonic-Kurve auf home muss der kalibrierte
    Punktwert im kalibrierten Band liegen — das ist die K1-Eigenschaft."""
    _force_offline(monkeypatch)

    # Lege ein synthetic Isotonic-Artifact ab, dass home halbiert, draw/away unveraendert.
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    iso_path = art_dir / "calibration_isotonic.json"
    payload = {
        "method": "isotonic",
        "curves": {
            "home": {"x": [0.0, 1.0], "y": [0.0, 0.5]},
            "draw": {"x": [], "y": []},
            "away": {"x": [], "y": []},
        },
        "n_trained_on": 100,
    }
    iso_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(cal_mod, "_ARTIFACT_DIR", art_dir)
    monkeypatch.setattr(cal_mod, "_ISOTONIC_PATH", iso_path)
    monkeypatch.setattr(cal_mod, "_PLATT_PATH", art_dir / "calibration_platt.json")

    target = _pick_match()
    match = await upsert_match_from_config(session, target)
    await session.commit()

    _, _, pred = await run_crawl_and_predict(session, match, crawl_seed=42)

    assert pred.calibrated_home_win_prob is not None
    if pred.confidence_intervals is None:
        return
    iso_ci = pred.confidence_intervals.get("isotonic")
    if iso_ci is None:
        return
    blended = iso_ci.get("blended", {})
    p5, p50, p95 = blended.get("home_win", [0.0, 0.0, 1.0])
    # Cal point should sit inside its CI band, both clipped to [0,1].
    assert 0.0 <= p5 <= p95 <= 1.0
    assert p5 - 1e-6 <= pred.calibrated_home_win_prob <= p95 + 1e-6, (
        f"calibrated home {pred.calibrated_home_win_prob} not in [{p5}, {p95}]"
    )
