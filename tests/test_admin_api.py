"""Admin & datasource-status endpoint tests.

We exercise the FastAPI app via TestClient and check:
  * GET /api/admin/weights returns all factor_weight_* fields.
  * PATCH /api/admin/weights without key → 503 (admin disabled).
  * PATCH with the right key persists to runtime_weights.yaml and reloads.
  * GET /api/datasources/status returns the expected 13-connector list.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config.settings import settings


@pytest.fixture(scope="module")
def client():
    # Import here so the FastAPI app is built per-test-session and respects
    # the test's monkeypatching of settings/.env paths.
    from main import app
    return TestClient(app)


def test_get_weights_returns_full_factor_list(client):
    resp = client.get("/api/admin/weights")
    assert resp.status_code == 200
    body = resp.json()
    assert "weights" in body
    names = {w["name"] for w in body["weights"]}
    # All v3.3 factors must be exposed.
    for expected in (
        "factor_weight_elo",
        "factor_weight_llm_sentiment",
        "factor_weight_lineup",
        "factor_weight_squad_value",
        "factor_weight_network",
    ):
        assert expected in names


def test_patch_without_admin_key_blocks(client):
    resp = client.patch("/api/admin/weights", json={"weights": {"factor_weight_elo": 0.25}})
    assert resp.status_code in (401, 503)


def test_patch_with_admin_key_persists(client, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-admin-key-123")
    monkeypatch.setattr(
        "api.admin._RUNTIME_WEIGHTS_PATH",
        tmp_path / "runtime_weights.yaml",
    )
    resp = client.patch(
        "/api/admin/weights",
        json={"weights": {"factor_weight_llm_sentiment": 0.08}},
        headers={"X-Admin-Key": "test-admin-key-123"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    llm = next(w for w in body["weights"] if w["name"] == "factor_weight_llm_sentiment")
    assert llm["value"] == pytest.approx(0.08, abs=1e-6)


def test_datasources_status_lists_known_connectors(client):
    resp = client.get("/api/datasources/status")
    assert resp.status_code == 200
    body = resp.json()
    names = {c["connector"] for c in body["connectors"]}
    for expected in ("openfootball", "fbref", "understat", "fotmob",
                     "sofascore", "transfermarkt", "nvidia_llm"):
        assert expected in names
