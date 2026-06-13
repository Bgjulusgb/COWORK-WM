"""Provenance summary — how much of a `mode: live` run is substantively live."""
from __future__ import annotations

from wm2026.pipeline import _provenance_summary


def test_summary_counts_every_mode_and_computes_coverage():
    prov = {
        "history_home":  {"mode": "live"},
        "history_away":  {"mode": "live"},
        "h2h":           {"mode": "cache"},
        "weather":       {"mode": "mock"},
        "fbref":         {"mode": "error"},
        "rss":           {"mode": "live"},
        "odds_api":      {"mode": "mock"},
        "research_slot": {"mode": "research"},
    }
    out = _provenance_summary(prov)
    assert out["total_slices"] == 8
    assert out["modes"]["live"] == 3
    assert out["modes"]["mock"] == 2
    assert out["modes"]["error"] == 1
    assert out["modes"]["cache"] == 1
    assert out["modes"]["research"] == 1
    # live + cache + research = 5/8 = 62.5%
    assert out["live_slices"] == 5
    assert out["live_coverage_pct"] == 62.5


def test_summary_empty_provenance_returns_zero_coverage():
    out = _provenance_summary({})
    assert out["total_slices"] == 0
    assert out["live_coverage_pct"] == 0.0
    assert out["live_slices"] == 0


def test_summary_handles_string_mode_values():
    """The compact JSON path replaces dict entries with their bare mode
    string — the summary helper must accept both shapes."""
    prov = {"history": "live", "weather": "mock"}
    out = _provenance_summary(prov)
    assert out["total_slices"] == 2
    assert out["live_slices"] == 1
    assert out["live_coverage_pct"] == 50.0


def test_summary_surfaces_in_report_json():
    """The Phase-8 build_report passes the summary through to the JSON."""
    import asyncio

    from wm2026.context import synth_config
    from wm2026.pipeline import run_prediction
    from wm2026.report import build_report

    cfg = synth_config(home_team="A", away_team="B")
    result = asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=0))
    js = build_report(result)["json"]
    assert "data_provenance_summary" in js
    s = js["data_provenance_summary"]
    assert s["total_slices"] > 0
    # Mock-mode → 0% live coverage (every slice is mock or error).
    assert s["live_coverage_pct"] == 0.0
