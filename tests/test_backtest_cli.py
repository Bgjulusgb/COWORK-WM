"""Tests for wm2026.backtest — score reports against actual outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wm2026.backtest import format_briefing, read_truth_csv, run_backtest


def _write_report(tmp: Path, match_id: str, p_1x2: tuple[float, float, float],
                  best_value: dict | None = None,
                  best_value_cons: dict | None = None) -> None:
    """Minimal report JSON with just the keys backtest reads."""
    payload = {
        "schema_version": "1.3",
        "match_id": match_id,
        "markets": {
            "1x2": {"home": p_1x2[0], "draw": p_1x2[1], "away": p_1x2[2]},
        },
        "best_value": best_value,
        "best_value_cons": best_value_cons,
    }
    (tmp / f"{match_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_truth(tmp: Path, rows: list[tuple[str, int, int]]) -> Path:
    p = tmp / "truth.csv"
    L = ["match_id,home_score,away_score"]
    for mid, hs, as_ in rows:
        L.append(f"{mid},{hs},{as_}")
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ── truth-CSV parsing ─────────────────────────────────────────────────────────
def test_read_truth_csv_skips_malformed_rows(tmp_path):
    csv_p = tmp_path / "truth.csv"
    csv_p.write_text(
        "match_id,home_score,away_score\nm1,2,1\nm2,,\nm3,1,X\nm4,0,0\n",
        encoding="utf-8",
    )
    assert read_truth_csv(csv_p) == {"m1": (2, 1), "m4": (0, 0)}


def test_read_truth_csv_rejects_missing_columns(tmp_path):
    csv_p = tmp_path / "bad.csv"
    csv_p.write_text("match_id,home_score\nm1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_truth_csv(csv_p)


def test_read_truth_csv_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_truth_csv(tmp_path / "nope.csv")


# ── core backtest ─────────────────────────────────────────────────────────────
def test_backtest_aggregates_brier_rps_and_pick_hits(tmp_path):
    """Two reports: model right on m1, wrong on m2; PnL nets to zero."""
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_report(
        reports, "m1", (0.70, 0.20, 0.10),
        best_value={"market": "1X2", "selection": "Home",
                    "decimal_odd": 2.00, "half_kelly_pct": 5.0},
        best_value_cons={"market": "1X2", "selection": "Home",
                         "decimal_odd": 2.00, "half_kelly_cons": 4.0},
    )
    _write_report(
        reports, "m2", (0.10, 0.20, 0.70),
        best_value={"market": "1X2", "selection": "Away",
                    "decimal_odd": 2.00, "half_kelly_pct": 5.0},
        best_value_cons=None,
    )
    truth = _write_truth(tmp_path, [("m1", 2, 1), ("m2", 2, 0)])
    rep = run_backtest(reports_dir=reports, truth_csv=truth, bankroll=1000.0)

    assert rep["n_evaluated"] == 2
    assert rep["best_value"]["attempts"] == 2
    assert rep["best_value"]["hits"] == 1
    assert rep["best_value"]["hit_rate"] == 0.5
    # Raw PnL: m1 home wins @ 2.00 stake 5% of 1000 = 50 → +50; m2 away loses → -50.
    assert abs(rep["best_value"]["pnl"]) < 1e-6
    # Conservative had only 1 attempt; it hit → stake 4% of 1000 = 40 @ odd 2.0 → +40.
    assert rep["best_value_cons"]["attempts"] == 1
    assert rep["best_value_cons"]["hits"] == 1
    assert abs(rep["best_value_cons"]["pnl"] - 40.0) < 1e-6


def test_backtest_counts_missing_truth_rows(tmp_path):
    """Reports without a matching truth row are reported as n_missing_truth."""
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_report(reports, "m1", (0.5, 0.3, 0.2))
    _write_report(reports, "m2_unmatched", (0.5, 0.3, 0.2))
    truth = _write_truth(tmp_path, [("m1", 1, 1)])
    rep = run_backtest(reports_dir=reports, truth_csv=truth, bankroll=1000.0)
    assert rep["n_evaluated"] == 1
    assert rep["n_missing_truth"] == 1


def test_backtest_handles_btts_and_double_chance_picks(tmp_path):
    """Hit-detection works for BTTS and Double-Chance selections."""
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_report(
        reports, "m1", (0.5, 0.3, 0.2),
        best_value={"market": "BTTS", "selection": "Yes",
                    "decimal_odd": 1.80, "half_kelly_pct": 3.0},
        best_value_cons={"market": "Double Chance", "selection": "1X",
                         "decimal_odd": 1.25, "half_kelly_cons": 2.0},
    )
    # 2-1 → BTTS Yes ✅, 1X (home or draw) ✅
    truth = _write_truth(tmp_path, [("m1", 2, 1)])
    rep = run_backtest(reports_dir=reports, truth_csv=truth, bankroll=1000.0)
    assert rep["best_value"]["hits"] == 1
    assert rep["best_value_cons"]["hits"] == 1


def test_format_briefing_renders_metrics(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_report(reports, "m1", (0.70, 0.20, 0.10))
    truth = _write_truth(tmp_path, [("m1", 2, 1)])
    rep = run_backtest(reports_dir=reports, truth_csv=truth, bankroll=1000.0)
    md = format_briefing(rep)
    assert "Brier" in md
    assert "RPS" in md
    assert "best_value" in md
    assert "1000" in md  # bankroll surfaces in the header
