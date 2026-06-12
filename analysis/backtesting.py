"""EXTEND-10: prediction-accuracy backtesting metrics.

Computes:
  * Brier score (lower = better; perfect = 0)
  * Log loss
  * RPS — Ranked Probability Score (the proper, *order-aware* 1X2 metric;
    Constantinou & Fenton 2012). Penalises being wrong by the ordinal distance
    Home > Draw > Away, so predicting Draw when Home happens hurts less than
    predicting Away. Lower = better; perfect = 0, worst = 1.
  * Calibration buckets ("70% predictions hit X% of the time")

The endpoint that exposes this lives in api/analytics.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CalibrationBucket:
    bucket_lo: float
    bucket_hi: float
    n: int
    mean_predicted: float
    mean_actual: float


@dataclass
class BacktestReport:
    n_evaluated: int = 0
    accuracy: float = 0.0
    brier: float = 0.0
    log_loss: float = 0.0
    rps: float = 0.0
    calibration: list[CalibrationBucket] = field(default_factory=list)


def _outcome_vec(home_score: int, away_score: int) -> tuple[float, float, float]:
    if home_score > away_score:
        return 1.0, 0.0, 0.0
    if home_score < away_score:
        return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


def _brier(p: tuple[float, float, float], y: tuple[float, float, float]) -> float:
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y))


def _log_loss(p: tuple[float, float, float], y: tuple[float, float, float]) -> float:
    # Add tiny epsilon so log(0) doesn't blow up.
    eps = 1e-12
    return -sum(yi * math.log(max(pi, eps)) for pi, yi in zip(p, y))


def _rps(p: tuple[float, float, float], y: tuple[float, float, float]) -> float:
    """Ranked Probability Score for the ordered outcome Home > Draw > Away.

    With r=3 ordered categories, RPS = 1/(r-1) · Σ_{k=1}^{r-1} (CP_k − CY_k)²
    over the *cumulative* predicted / actual vectors. 0 = perfect, 1 = worst
    (all mass on the category at the opposite end of the order).
    """
    cp = 0.0
    cy = 0.0
    acc = 0.0
    for k in range(len(p) - 1):       # only the first r-1 cumulatives matter
        cp += p[k]
        cy += y[k]
        acc += (cp - cy) ** 2
    return acc / (len(p) - 1)


def compute(rows: Iterable) -> BacktestReport:
    """`rows` are MatchPrediction ORM rows with actual_home_score / actual_away_score set."""
    n = 0
    correct = 0
    brier_sum = 0.0
    log_sum = 0.0
    rps_sum = 0.0
    buckets: dict[int, list[tuple[float, float]]] = {i: [] for i in range(10)}

    for r in rows:
        if r.actual_home_score is None or r.actual_away_score is None:
            continue
        p_vec = (float(r.home_win_prob or 0), float(r.draw_prob or 0), float(r.away_win_prob or 0))
        y_vec = _outcome_vec(int(r.actual_home_score), int(r.actual_away_score))
        brier_sum += _brier(p_vec, y_vec)
        log_sum += _log_loss(p_vec, y_vec)
        rps_sum += _rps(p_vec, y_vec)

        predicted_idx = max(range(3), key=lambda i: p_vec[i])
        actual_idx = max(range(3), key=lambda i: y_vec[i])
        if predicted_idx == actual_idx:
            correct += 1

        predicted_prob = p_vec[predicted_idx]
        bucket_idx = min(9, int(predicted_prob * 10))
        buckets[bucket_idx].append((predicted_prob, 1.0 if predicted_idx == actual_idx else 0.0))
        n += 1

    if n == 0:
        return BacktestReport()

    calibration = [
        CalibrationBucket(
            bucket_lo=i / 10.0,
            bucket_hi=(i + 1) / 10.0,
            n=len(items),
            mean_predicted=sum(p for p, _ in items) / len(items) if items else 0.0,
            mean_actual=sum(a for _, a in items) / len(items) if items else 0.0,
        )
        for i, items in buckets.items()
    ]

    return BacktestReport(
        n_evaluated=n,
        accuracy=correct / n,
        brier=brier_sum / n,
        log_loss=log_sum / n,
        rps=rps_sum / n,
        calibration=[c for c in calibration if c.n > 0],
    )
