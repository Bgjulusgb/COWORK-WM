"""Tests for the sklearn-free calibration path + market anchoring + offline fit.

These exercise the pure-Python PAV isotonic and Newton Platt fallbacks (sklearn
is an optional extra, absent in CI), the market-anchor calibration, and the
offline CSV fit script.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import calibration as calib
from analysis.calibration import (
    _isotonic_pav_curve,
    _pav,
    _platt_newton,
    fit_calibrators,
    market_anchor,
)


# ── PAV isotonic (pure) ───────────────────────────────────────────────────────
def test_pav_output_is_non_decreasing():
    out = _pav([0.9, 0.1, 0.8, 0.2, 0.95])
    assert out == sorted(out)                 # isotonic ⇒ non-decreasing
    assert len(out) == 5
    # known pooling: [0.9,0.1] → 0.5 each; [0.8,0.2] → 0.5; all ≤ 0.95
    assert out[-1] == pytest.approx(0.95)


def test_isotonic_pav_curve_tames_overconfidence():
    # 0.8-bucket only wins 6/10, 0.2-bucket only wins 1/10 → the curve must pull
    # the over-confident 0.8 prediction down toward its realised 0.6 rate.
    xs = [0.2] * 10 + [0.8] * 10
    ys = ([1] + [0] * 9) + ([1] * 6 + [0] * 4)
    curve = _isotonic_pav_curve(xs, ys)
    assert curve.transform(0.8) == pytest.approx(0.6, abs=1e-9)
    assert curve.transform(0.2) == pytest.approx(0.1, abs=1e-9)
    assert curve.transform(0.2) <= curve.transform(0.8)      # monotone


# ── Platt (pure Newton) ───────────────────────────────────────────────────────
def test_platt_newton_recovers_positive_slope():
    # Label correlates with p → fitted logistic must increase in p.
    pairs_p = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
    pairs_y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    pc = _platt_newton(pairs_p, pairs_y)
    assert pc.a > 0.0
    assert pc.transform(0.8) > pc.transform(0.2)


def test_platt_newton_degenerate_single_class_is_identity():
    pc = _platt_newton([0.3, 0.5, 0.7, 0.9, 0.4], [1, 1, 1, 1, 1])
    assert pc.a == 1.0 and pc.b == 0.0


# ── market anchor ─────────────────────────────────────────────────────────────
def test_market_anchor_blends_and_renormalises():
    model = (0.50, 0.25, 0.25)
    market = (0.40, 0.30, 0.30)
    half = market_anchor(*model, market, weight=0.5)
    assert half is not None
    assert sum(half.values()) == pytest.approx(1.0)
    assert half["home"] == pytest.approx(0.45)       # midpoint of 0.50 and 0.40

    pure_model = market_anchor(*model, market, weight=0.0)
    assert pure_model["home"] == pytest.approx(0.50)
    pure_market = market_anchor(*model, market, weight=1.0)
    assert pure_market["home"] == pytest.approx(0.40)


def test_market_anchor_rejects_bad_market():
    assert market_anchor(0.5, 0.3, 0.2, None) is None
    assert market_anchor(0.5, 0.3, 0.2, [0.5, 0.5]) is None      # too short
    assert market_anchor(0.5, 0.3, 0.2, (0.0, 0.0, 0.0)) is None  # degenerate


# ── fit_calibrators works WITHOUT sklearn (core deps) ─────────────────────────
def _rows(spec):
    return [
        SimpleNamespace(home_win_prob=h, draw_prob=d, away_win_prob=a,
                        actual_home_score=hs, actual_away_score=as_)
        for (h, d, a, hs, as_) in spec
    ]


def test_fit_calibrators_pure_python_yields_nonempty_isotonic(tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(calib, "_ISOTONIC_PATH", tmp_path / "iso.json")
    monkeypatch.setattr(calib, "_PLATT_PATH", tmp_path / "platt.json")
    rows = _rows([
        (0.70, 0.20, 0.10, 2, 0), (0.65, 0.20, 0.15, 1, 0),
        (0.55, 0.25, 0.20, 0, 0), (0.40, 0.35, 0.25, 1, 1),
        (0.30, 0.30, 0.40, 0, 2), (0.45, 0.30, 0.25, 0, 1),
        (0.60, 0.25, 0.15, 3, 1), (0.50, 0.25, 0.25, 1, 2),
        (0.35, 0.40, 0.25, 1, 1), (0.75, 0.15, 0.10, 2, 0),
    ])
    iso, platt = fit_calibrators(rows)
    # The pure-Python fallback must produce a real (non-empty) isotonic curve,
    # not the empty identity the old sklearn-missing branch returned.
    assert len(iso.curves["home"].x) >= 2
    assert (tmp_path / "iso.json").exists()


# ── offline CSV loader ────────────────────────────────────────────────────────
def test_rows_from_csv_parses_and_skips_incomplete(tmp_path):
    from scripts.fit_calibration_offline import rows_from_csv

    csv_path = tmp_path / "hist.csv"
    csv_path.write_text(
        "home_win_prob,draw_prob,away_win_prob,home_score,away_score\n"
        "0.62,0.24,0.14,2,1\n"
        "0.31,0.30,0.39,0,0\n"
        "0.50,0.25,0.25,,\n"             # incomplete → skipped
        "0.40,0.30,0.30,1,2\n",
        encoding="utf-8",
    )
    rows = rows_from_csv(csv_path)
    assert len(rows) == 3
    assert rows[0].actual_home_score == 2 and rows[0].actual_away_score == 1
    assert isinstance(rows[0].home_win_prob, float)


def test_rows_from_csv_missing_columns_raises(tmp_path):
    from scripts.fit_calibration_offline import rows_from_csv

    bad = tmp_path / "bad.csv"
    bad.write_text("home_win_prob,draw_prob\n0.5,0.3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rows_from_csv(bad)
