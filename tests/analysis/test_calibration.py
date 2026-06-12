"""Tests für analysis.calibration — Isotonic + Platt-Curve, fit/apply,
und v3.7-Neu: transform_intervals (kalibrierte Bootstrap-CIs)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import calibration
from analysis.calibration import (
    CalibrationArtifact,
    IsotonicCurve,
    PlattCurve,
    apply,
    fit_calibrators,
    transform_intervals,
)


# ── Kurven ────────────────────────────────────────────────────────────────────


def test_isotonic_curve_is_identity_when_empty():
    iso = IsotonicCurve()
    assert iso.transform(0.0) == 0.0
    assert iso.transform(0.5) == 0.5
    assert iso.transform(1.0) == 1.0


def test_isotonic_curve_interpolates_linearly():
    iso = IsotonicCurve(x=[0.0, 0.5, 1.0], y=[0.1, 0.5, 0.9])
    assert iso.transform(0.25) == pytest.approx(0.3)   # halfway between 0.1 and 0.5
    assert iso.transform(0.75) == pytest.approx(0.7)   # halfway between 0.5 and 0.9


def test_isotonic_curve_clips_out_of_bounds():
    iso = IsotonicCurve(x=[0.2, 0.8], y=[0.3, 0.7])
    assert iso.transform(-0.1) == 0.3   # left clip
    assert iso.transform(1.5) == 0.7    # right clip


def test_platt_curve_identity_when_a1_b0():
    """a=1, b=0 ⇒ sigmoid(logit(p)) == p."""
    pc = PlattCurve(a=1.0, b=0.0)
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert pc.transform(p) == pytest.approx(p, abs=1e-6)


def test_platt_curve_with_b_only_shifts():
    pc = PlattCurve(a=1.0, b=1.0)
    # sigmoid(0 + 1) = 0.7310585
    assert pc.transform(0.5) == pytest.approx(1 / (1 + math.exp(-1.0)), abs=1e-6)


# ── apply (Hot-Path) ──────────────────────────────────────────────────────────


def test_apply_returns_none_when_artifact_is_none():
    assert apply(None, 0.4, 0.3, 0.3) is None


def test_apply_renormalises_to_sum_one():
    iso = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 0.5]),  # halves home
            "draw": IsotonicCurve(),
            "away": IsotonicCurve(),
        },
    )
    out = apply(iso, 0.6, 0.2, 0.2)
    assert out is not None
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["home"] < 0.6  # halved before renorm


# ── fit_calibrators ───────────────────────────────────────────────────────────


def _rows(probs_and_outcomes):
    """Tiny stand-in für MatchPrediction rows."""
    return [
        SimpleNamespace(
            home_win_prob=p_home,
            draw_prob=p_draw,
            away_win_prob=p_away,
            actual_home_score=hs,
            actual_away_score=as_,
        )
        for (p_home, p_draw, p_away, hs, as_) in probs_and_outcomes
    ]


def test_fit_calibrators_writes_both_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(calibration, "_ISOTONIC_PATH", tmp_path / "iso.json")
    monkeypatch.setattr(calibration, "_PLATT_PATH", tmp_path / "platt.json")

    rows = _rows([
        (0.7, 0.2, 0.1, 2, 0),  # home actually wins
        (0.6, 0.2, 0.2, 1, 0),  # home wins
        (0.5, 0.3, 0.2, 0, 0),  # draw
        (0.3, 0.4, 0.3, 1, 1),  # draw
        (0.2, 0.3, 0.5, 0, 2),  # away wins
        (0.4, 0.3, 0.3, 0, 1),  # away wins
        (0.6, 0.2, 0.2, 2, 1),  # home wins
        (0.5, 0.2, 0.3, 1, 2),  # away wins
        (0.4, 0.4, 0.2, 1, 1),  # draw
        (0.7, 0.2, 0.1, 3, 0),  # home wins
    ])
    iso, platt = fit_calibrators(rows)
    assert iso.method == "isotonic"
    assert platt.method == "platt"
    assert (tmp_path / "iso.json").exists()
    assert (tmp_path / "platt.json").exists()
    # Both artifacts should at least include the home outcome curve.
    assert "home" in iso.curves
    assert "home" in platt.curves


# ── transform_intervals (K1) ──────────────────────────────────────────────────


def test_transform_intervals_returns_none_when_artifact_missing():
    ci = {"blended": {"home_win": [0.1, 0.2, 0.3]}}
    assert transform_intervals(ci, None) is None
    assert transform_intervals(None, CalibrationArtifact(method="isotonic", curves={})) is None


def test_transform_intervals_identity_with_empty_curves_renormalises_each_quantile():
    # Empty curves act as identity per outcome, but the helper still renormalises
    # each 1X2 quantile-triple to Σ=1 (BEFORE the final per-outcome sort).
    # We probe this via input that already sums to 1 → identity + renorm + sort
    # round-trip equals the input.
    ci = {
        "blended": {
            "home_win": [0.4, 0.5, 0.6],
            "draw":     [0.4, 0.3, 0.2],
            "away_win": [0.2, 0.2, 0.2],
            "over_25":  [0.45, 0.55, 0.65],
        }
    }
    art = CalibrationArtifact(
        method="isotonic",
        curves={"home": IsotonicCurve(), "draw": IsotonicCurve(), "away": IsotonicCurve()},
    )
    out = transform_intervals(ci, art)
    assert out is not None
    # Each input column already sums to 1.0, identity transform preserves them,
    # renorm is a no-op → outputs equal sorted inputs.
    assert out["blended"]["home_win"] == pytest.approx([0.4, 0.5, 0.6])
    assert out["blended"]["draw"] == pytest.approx([0.2, 0.3, 0.4])
    assert out["blended"]["away_win"] == pytest.approx([0.2, 0.2, 0.2])
    # Bernoulli markets pass through unchanged (no renorm).
    assert out["blended"]["over_25"] == pytest.approx([0.45, 0.55, 0.65])


def test_transform_intervals_keeps_monotonicity_even_with_steep_curves():
    # After renorm, a steep home-curve could break p5<=p50<=p95 monotonicity
    # for a single outcome. The helper sorts each triple so the UI still
    # gets a well-defined band.
    ci = {
        "blended": {
            "home_win": [0.30, 0.50, 0.70],
            "draw":     [0.20, 0.25, 0.30],
            "away_win": [0.10, 0.15, 0.20],
        }
    }
    art = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=[0.0, 0.5, 1.0], y=[0.0, 0.4, 0.9]),  # convex
            "draw": IsotonicCurve(x=[0.0, 1.0], y=[0.05, 0.95]),
            "away": IsotonicCurve(x=[0.0, 1.0], y=[0.05, 0.95]),
        },
    )
    out = transform_intervals(ci, art)
    assert out is not None
    for key in ("home_win", "draw", "away_win"):
        p5, p50, p95 = out["blended"][key]
        assert p5 <= p50 <= p95, f"{key} not monotonic: {p5} {p50} {p95}"


def test_transform_intervals_brackets_calibrated_point_property_K1():
    """The K1 promise: the calibrated CI band brackets the calibrated point.

    Use a steep (sqrt-like) isotonic curve — that's the exact shape where the
    no-renorm v3.7 design FAILED: ``apply()`` Σ=1-renormalises the point, so a
    CI that wasn't renormalised landed entirely above the point. With per-
    quantile renorm in :func:`transform_intervals` the property holds.
    """
    raw_home, raw_draw, raw_away = 0.4946, 0.2304, 0.2750
    ci = {
        "blended": {
            "home_win": [0.3774, raw_home, 0.6010],
            "draw":     [0.1995, raw_draw, 0.2622],
            "away_win": [0.1915, raw_away, 0.3728],
        }
    }
    # sqrt-like curve (over-confident dampening): y = 0.92*sqrt(x) + 0.04
    xs = [i / 14 for i in range(15)]
    ys = [0.92 * (x ** 0.5) + 0.04 for x in xs]
    art = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=xs, y=ys),
            "draw": IsotonicCurve(x=xs, y=ys),
            "away": IsotonicCurve(x=xs, y=ys),
        },
    )

    cal_pt = apply(art, raw_home, raw_draw, raw_away)
    assert cal_pt is not None
    out = transform_intervals(ci, art)
    assert out is not None

    for outcome_key, cal_key in (("home_win", "home"), ("draw", "draw"), ("away_win", "away")):
        p5, p50, p95 = out["blended"][outcome_key]
        point = cal_pt[cal_key]
        assert p5 <= point <= p95, (
            f"K1 violated for {outcome_key}: p5={p5:.4f} > point={point:.4f} > p95={p95:.4f}"
        )


def test_transform_intervals_halving_curve_pulls_home_down():
    ci = {
        "blended": {
            "home_win": [0.4, 0.5, 0.6],
            "draw":     [0.3, 0.3, 0.3],
            "away_win": [0.2, 0.2, 0.2],
        }
    }
    art = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 0.5]),  # halve home
            "draw": IsotonicCurve(),
            "away": IsotonicCurve(),
        },
    )
    out = transform_intervals(ci, art)
    assert out is not None
    # Halving home + renorm pulls home below its raw values, lifts draw/away.
    # K1 invariants: monotonic + in [0,1]. Per-outcome sort means Σ=1 is NOT
    # preserved per quantile (sort across quantiles within an outcome shuffles
    # values relative to the cross-outcome renorm), but the band-brackets-point
    # K1 property is preserved (tested separately).
    for key in ("home_win", "draw", "away_win"):
        p5, p50, p95 = out["blended"][key]
        assert 0.0 <= p5 <= p50 <= p95 <= 1.0
    # home is now lower than draw/away at every quantile (halving dominates renorm).
    for q in range(3):
        assert out["blended"]["home_win"][q] < out["blended"]["draw"][q]


def test_transform_intervals_processes_all_model_keys():
    ci = {
        "blended":     {"home_win": [0.4, 0.5, 0.6], "draw": [0.2, 0.3, 0.4], "away_win": [0.2, 0.2, 0.2]},
        "poisson":     {"home_win": [0.3, 0.5, 0.7], "draw": [0.2, 0.2, 0.2], "away_win": [0.2, 0.3, 0.4]},
        "negbin":      {"home_win": [0.3, 0.4, 0.5], "draw": [0.2, 0.3, 0.4], "away_win": [0.2, 0.3, 0.4]},
        "glm_poisson": {"home_win": [0.4, 0.5, 0.6], "draw": [0.2, 0.3, 0.4], "away_win": [0.2, 0.2, 0.2]},
    }
    art = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 0.5]),
            "draw": IsotonicCurve(),
            "away": IsotonicCurve(),
        },
    )
    out = transform_intervals(ci, art)
    assert out is not None
    assert set(out.keys()) == set(ci.keys())
    for model in ci.keys():
        for key in ("home_win", "draw", "away_win"):
            p5, p50, p95 = out[model][key]
            assert 0.0 <= p5 <= p50 <= p95 <= 1.0


def test_transform_intervals_handles_zero_collapse_gracefully():
    """All curves collapse to 0 → per-quantile Σ=0 → fall back to uniform (1/3 each)
    instead of dividing by zero. The output remains finite and in [0,1]."""
    ci = {"blended": {"home_win": [0.4, 0.5, 0.6], "draw": [0.2, 0.3, 0.4], "away_win": [0.1, 0.2, 0.3]}}
    art = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 0.0]),
            "draw": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 0.0]),
            "away": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 0.0]),
        },
    )
    out = transform_intervals(ci, art)
    assert out is not None
    for key in ("home_win", "draw", "away_win"):
        for v in out["blended"][key]:
            assert math.isfinite(v)
            assert v == pytest.approx(1.0 / 3.0)


def test_transform_intervals_clips_to_unit_interval():
    # p95 > 1.0 simulates a numeric edge case. Curve is identity, so we expect
    # the output triple to be clipped to [0,1].
    ci = {"blended": {
        "home_win": [0.9, 0.95, 1.05],
        "draw":     [0.05, 0.04, 0.0],     # NB: input order not monotonic
        "away_win": [0.05, 0.01, -0.05],   # NB: negative numeric edge
    }}
    art = CalibrationArtifact(
        method="isotonic",
        curves={
            "home": IsotonicCurve(x=[0.0, 1.0], y=[0.0, 1.0]),
            "draw": IsotonicCurve(),
            "away": IsotonicCurve(),
        },
    )
    out = transform_intervals(ci, art)
    assert out is not None
    for key in ("home_win", "draw", "away_win"):
        for v in out["blended"][key]:
            assert 0.0 <= v <= 1.0, f"{key} value {v} out of [0,1]"
