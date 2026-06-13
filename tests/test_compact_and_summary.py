"""Token-budget pipeline: compact JSON, wm2026 summary, wm2026 doctor.

Bare pytest (uses ``asyncio.run`` directly per repo convention).
"""
from __future__ import annotations

import asyncio
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wm2026.context import synth_config
from wm2026.pipeline import run_prediction
from wm2026.report import (
    _compress_derived,
    _compress_edge_table,
    _compress_factors,
    _compress_provenance,
    build_report,
    compact,
)
from wm2026.summary import summarise


def _run(cfg, **kw):
    return asyncio.run(run_prediction(cfg, mode="mock", bootstrap_n=48, **kw))


# ── compact() shape contract ──────────────────────────────────────────────────
def test_compact_keeps_betting_shape_and_signals_itself():
    cfg = synth_config(home_team="Brazil", away_team="Serbia",
                       home_xg=1.9, away_xg=0.9)
    result = _run(cfg, odds_1x2=[2.10, 3.40, 3.20],
                  odds_ou25=[1.85, 1.95], odds_btts=[1.80, 2.00])
    js = build_report(result)["json"]
    c = compact(js)
    assert c["compact"] is True
    for k in ("match_id", "lambda_home", "lambda_away", "markets",
              "derived_markets", "edge_table",
              "best_value_cons", "ensemble_confidence",
              "confidence_intervals", "warnings"):
        assert k in c, f"compact lost {k}"
    # Per-model markets + correct-score top 5 are debug-only → dropped.
    assert "per_model" not in c
    assert "correct_score_top5" not in c["markets"]


def test_compact_reduces_size_substantially():
    cfg = synth_config(home_team="Italy", away_team="Ghana")
    result = _run(cfg, odds_1x2=[1.80, 3.50, 4.50])
    js = build_report(result)["json"]
    full = len(json.dumps(js))
    slim = len(json.dumps(compact(js)))
    assert slim < full
    # Real-world reduction is 30–55 %; pin a conservative floor so a future
    # change can't quietly break the token contract.
    assert (full - slim) / full > 0.25, f"compact only saved {(full-slim)/full:.0%}"


def test_compact_drops_unavailable_factors_and_raw_data():
    cfg = synth_config(home_team="A", away_team="B")
    result = _run(cfg)
    js = build_report(result)["json"]
    c = compact(js)
    raw_factor_count = len(js["factors"])
    kept = c["factors"]
    assert 0 < len(kept) <= raw_factor_count
    assert all(f.get("available") is True for f in kept)
    for f in kept:
        assert "raw_data" not in f
        assert "cached_at" not in f


def test_compact_data_sources_collapse_to_mode_tokens():
    cfg = synth_config(home_team="A", away_team="B")
    result = _run(cfg)
    js = build_report(result)["json"]
    c = compact(js)
    assert c["data_sources"]                 # non-empty
    sample = next(iter(c["data_sources"].values()))
    # The full report has a dict per slice; compact replaces it with a string.
    assert isinstance(sample, str)
    assert sample in {"mock", "live", "cache", "error", "research", "?"}


def test_compact_ah_lines_default_to_five_main_lines():
    cfg = synth_config(home_team="A", away_team="B")
    result = _run(cfg)
    js = build_report(result)["json"]
    c = compact(js)
    ah = c["derived_markets"]["asian_handicap"]
    lines = sorted(r["line"] for r in ah)
    assert lines == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_compact_ah_lines_can_be_overridden():
    cfg = synth_config(home_team="A", away_team="B")
    result = _run(cfg)
    js = build_report(result)["json"]
    c = compact(js, ah_lines={-0.5, 0.5})
    ah = c["derived_markets"]["asian_handicap"]
    assert {r["line"] for r in ah} == {-0.5, 0.5}


def test_compact_edge_table_sorts_by_abs_edge_and_drops_no_odds():
    rows = [
        {"market": "1X2", "selection": "Home", "decimal_odd": 2.0, "edge_pct": +3.0},
        {"market": "1X2", "selection": "Draw", "decimal_odd": 3.4, "edge_pct": -8.0},
        {"market": "1X2", "selection": "Away", "decimal_odd": 3.2, "edge_pct": +1.0},
        {"market": "BTTS", "selection": "No",  "decimal_odd": None, "edge_pct": None},
    ]
    out = _compress_edge_table(rows, top_n=2)
    # Draw has the biggest |edge|, then Home; Away drops out at top_n=2.
    sels = [r["selection"] for r in out]
    assert sels == ["Draw", "Home"]
    # No odds → never appears (filtered out before sorting).
    assert all(r.get("decimal_odd") is not None for r in out)


def test_helpers_handle_empty_inputs():
    assert _compress_factors([]) == []
    assert _compress_provenance({}) == {}
    assert _compress_derived({}) == {}
    assert _compress_edge_table([]) == []


def test_pipeline_ah_lines_passthrough():
    cfg = synth_config(home_team="A", away_team="B")
    res = _run(cfg, ah_lines=[-0.5, 0.0, 0.5])
    ah = res["derived_markets"]["asian_handicap"]
    assert [r["line"] for r in ah] == [-0.5, 0.0, 0.5]


# ── summary() output contract ─────────────────────────────────────────────────
def test_summary_renders_required_blocks():
    cfg = synth_config(home_team="France", away_team="Senegal")
    result = _run(cfg, odds_1x2=[1.70, 3.60, 4.50],
                  odds_ou25=[1.85, 1.95], odds_btts=[1.80, 2.00],
                  bankroll=1000.0)
    js = build_report(result)["json"]
    text = summarise(js, top_edges=3)
    assert "## λ + CI" in text
    assert "## Edges" in text
    assert "## Recommendation" in text
    assert "France" in text and "Senegal" in text
    # Disclaimer is mandatory.
    assert "Forschung" in text or "research" in text.lower() or "Wett-Empfehlung" in text


def test_summary_is_compact_under_2k_chars():
    cfg = synth_config(home_team="A", away_team="B", home_xg=1.5, away_xg=1.2)
    result = _run(cfg, odds_1x2=[2.10, 3.40, 3.20])
    text = summarise(build_report(result)["json"])
    # ≈ 500 tokens budget. Pin a conservative ceiling.
    assert len(text) < 2500, f"summary grew to {len(text)} chars"


def test_summary_signals_pass_when_no_p5_survivor():
    js = {
        "fixture": {"home": "X", "away": "Y", "stage": "Group"},
        "mode": "mock", "ensemble_confidence": 0.5,
        "factors_used": 12, "factors_total": 20,
        "lambda_home": {"p5": 0.9, "p50": 1.2, "p95": 1.5},
        "lambda_away": {"p5": 0.8, "p50": 1.1, "p95": 1.4},
        "markets": {"1x2": {"home": 0.4, "draw": 0.25, "away": 0.35},
                    "over_under": {"over": 0.5, "under": 0.5},
                    "btts": {"yes": 0.5, "no": 0.5}},
        "confidence_intervals": {"blended": {}},
        "edge_table": [],
        "best_value": None, "best_value_cons": None,
        "warnings": [], "claude_tasks": [],
    }
    text = summarise(js)
    assert "Pass" in text and "p5" in text


def test_summary_handles_missing_fields_without_crash():
    text = summarise({})
    assert "Home" in text and "Away" in text
    assert "Pass" in text


def test_summary_picks_calibrated_1x2_when_present():
    js = {
        "fixture": {"home": "A", "away": "B"},
        "mode": "mock",
        "lambda_home": {"p5": 1, "p50": 1.3, "p95": 1.6},
        "lambda_away": {"p5": 1, "p50": 1.3, "p95": 1.6},
        "markets": {"1x2": {"home": 0.4, "draw": 0.25, "away": 0.35},
                    "over_under": {"over": 0.5, "under": 0.5},
                    "btts": {"yes": 0.5, "no": 0.5}},
        "calibration": {"applied": True,
                        "calibrated": {"home_win": 0.50, "draw": 0.20, "away_win": 0.30}},
        "confidence_intervals": {"blended": {}},
        "edge_table": [],
        "best_value": None, "best_value_cons": None,
        "ensemble_confidence": 0.7, "factors_used": 15, "factors_total": 20,
        "warnings": [], "claude_tasks": [],
    }
    text = summarise(js)
    assert "calibrated" in text
    assert "50.0%" in text and "30.0%" in text   # honors calibrated values


# ── doctor() module ───────────────────────────────────────────────────────────
def test_doctor_module_runs_clean(capsys):
    from wm2026.doctor import run
    code = run(verbose=False)
    out = capsys.readouterr().out
    assert code == 0
    assert "core deps ok" in out
    assert "pipeline modules import" in out
    assert "schema fields" in out
    assert "doctor: all checks passed" in out


def test_doctor_json_emits_status_dict(capsys):
    from wm2026.doctor import main
    code = main(["--json"])
    out = capsys.readouterr().out
    status = json.loads(out)
    assert code == 0
    assert status["core_missing"] == 0
    assert status["pipeline_missing"] == 0
    assert status["smoke_ok"] is True
    assert status["schema_version"] == "1.3"


# ── CLI plumbing (subprocess) ─────────────────────────────────────────────────
def _cli(*args, **kw):
    return subprocess.run([sys.executable, "-m", "wm2026.cli", *args],
                          capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parents[1], **kw)


def test_cli_doctor_exit_zero():
    r = _cli("doctor", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["smoke_ok"] is True


def test_cli_summary_command_reads_disk(tmp_path):
    r = _cli("predict", "--mode", "mock", "--home", "A", "--away", "B",
             "--odds", "2.10/3.40/3.20", "--format", "json",
             "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    json_path = next(tmp_path.glob("*.json"))
    summary_path = next(tmp_path.glob("*.summary.md"))
    assert summary_path.exists()
    r2 = _cli("summary", str(json_path))
    assert r2.returncode == 0, r2.stderr
    assert "Recommendation" in r2.stdout


def test_cli_summary_reads_gzip(tmp_path):
    r = _cli("predict", "--mode", "mock", "--home", "A", "--away", "B",
             "--odds", "2.10/3.40/3.20", "--gzip", "--format", "json",
             "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    gz = next(tmp_path.glob("*.json.gz"))
    # The gzip is a real gzip of the JSON.
    with gzip.open(gz, "rt", encoding="utf-8") as fh:
        js = json.load(fh)
    assert js["schema_version"] == "1.3"
    r2 = _cli("summary", str(gz))
    assert r2.returncode == 0
    assert "λ" in r2.stdout or "lambda" in r2.stdout.lower()


def test_cli_predict_compact_flag(tmp_path):
    r = _cli("predict", "--mode", "mock", "--home", "A", "--away", "B",
             "--odds", "2.10/3.40/3.20", "--compact", "--format", "json",
             "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    js = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert js.get("compact") is True
    assert "per_model" not in js


def test_cli_predict_ah_lines_negative_with_equals(tmp_path):
    """argparse requires the ``=`` form for any value starting with ``-``."""
    r = _cli("predict", "--mode", "mock", "--home", "A", "--away", "B",
             "--ah-lines=-1,-0.5,0,0.5,1", "--format", "json",
             "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    js = json.loads(next(tmp_path.glob("*.json")).read_text())
    ah = js["derived_markets"]["asian_handicap"]
    assert {r["line"] for r in ah} == {-1.0, -0.5, 0.0, 0.5, 1.0}


def test_cli_html_charts_external_shrinks_html(tmp_path):
    """External-charts mode references PNG siblings → tiny HTML."""
    pytest.importorskip("matplotlib")  # PNG charts need the [viz] extra
    r = _cli("predict", "--mode", "mock", "--home", "A", "--away", "B",
             "--odds", "2.10/3.40/3.20", "--format", "html",
             "--charts", "--charts-external", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    html = next(tmp_path.glob("*.html"))
    text = html.read_text(encoding="utf-8")
    assert "data:image/png;base64," not in text
    # External references point at the on-disk PNGs by match_id prefix.
    assert "_tornado.png" in text and "_heatmap.png" in text
    assert html.stat().st_size < 25_000     # was ~95 KB with embedded charts
    # The two PNGs should actually be on disk next to the HTML.
    assert next(tmp_path.glob("*_tornado.png")).exists()
    assert next(tmp_path.glob("*_heatmap.png")).exists()


def test_cli_format_summary_streams_briefing(tmp_path):
    r = _cli("predict", "--mode", "mock", "--home", "A", "--away", "B",
             "--odds", "2.10/3.40/3.20", "--format", "summary",
             "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "λ + CI" in r.stdout
    assert "Recommendation" in r.stdout
