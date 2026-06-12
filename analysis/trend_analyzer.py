"""Trend, volatility and anomaly detection on sentiment time series.

Inputs are sequences of `(timestamp, score)` pairs or the raw post objects
already used elsewhere in the pipeline.

* `linear_slope`     — OLS slope (sentiment change per hour, normalised to [-1, 1])
* `r_squared`        — quality of the linear fit
* `volatility`       — std-dev of bucket means (how jumpy is the conversation?)
* `detect_anomalies` — buckets that deviate > k·std from the running mean
* `compute_trend`    — top-level helper that turns a list of joined rows into a
                       `TrendReport` ready for the API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class TrendBucket:
    bucket_start: datetime
    n: int
    mean_score: float


@dataclass
class Anomaly:
    bucket_start: datetime
    delta: float       # score - running_mean
    z_score: float
    direction: str     # "spike" | "drop"


@dataclass
class TrendReport:
    slope_per_hour: float = 0.0
    slope_strength: float = 0.0        # |slope| scaled by r^2, ~0..1
    r_squared: float = 0.0
    volatility: float = 0.0
    momentum_24h: float = 0.0          # mean of last 24h minus mean of preceding 24h
    momentum_6h: float = 0.0
    direction: str = "flat"            # "rising" | "falling" | "flat"
    anomalies: List[Anomaly] = field(default_factory=list)
    buckets: List[TrendBucket] = field(default_factory=list)


def _to_score_series(posts, *, team: str) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    for p in posts:
        if p.team_attribution not in (team, "neutral"):
            continue
        ts = p.created_utc
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append((ts, float(p.ensemble_score or 0.0)))
    out.sort(key=lambda x: x[0])
    return out


def _bucketize(series: List[Tuple[datetime, float]], bucket_hours: int) -> List[TrendBucket]:
    if not series:
        return []
    start = series[0][0]
    bucket: Dict[datetime, List[float]] = {}
    delta = timedelta(hours=bucket_hours)
    for ts, score in series:
        # snap to bucket floor
        offset_h = (ts - start).total_seconds() / 3600.0
        b_idx = int(offset_h // bucket_hours)
        b_start = start + b_idx * delta
        bucket.setdefault(b_start, []).append(score)
    return [
        TrendBucket(bucket_start=k, n=len(v), mean_score=float(mean(v)))
        for k, v in sorted(bucket.items())
    ]


def _ols_slope(buckets: List[TrendBucket]) -> Tuple[float, float]:
    """Return (slope_per_hour, r_squared) using plain OLS on (hours, mean_score).

    Slope is then clipped to [-1, 1]: a 2.0 swing over 24 hours saturates
    interpretation, anything bigger is treated as 1.0 for the strength score.
    """
    if len(buckets) < 2:
        return 0.0, 0.0
    t0 = buckets[0].bucket_start
    xs = [(b.bucket_start - t0).total_seconds() / 3600.0 for b in buckets]
    ys = [b.mean_score for b in buckets]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return float(slope), float(max(0.0, min(1.0, r2)))


def _detect_anomalies(buckets: List[TrendBucket], k: float = 2.0) -> List[Anomaly]:
    if len(buckets) < 4:
        return []
    means = [b.mean_score for b in buckets]
    mu = mean(means)
    sigma = pstdev(means)
    if sigma == 0:
        return []
    out: List[Anomaly] = []
    for b in buckets:
        z = (b.mean_score - mu) / sigma
        if abs(z) >= k:
            out.append(
                Anomaly(
                    bucket_start=b.bucket_start,
                    delta=float(b.mean_score - mu),
                    z_score=float(z),
                    direction="spike" if z > 0 else "drop",
                )
            )
    return out


def _window_mean(series: List[Tuple[datetime, float]], hours: int,
                 *, now: Optional[datetime] = None) -> float:
    if not series:
        return 0.0
    now = now or datetime.now(timezone.utc)
    cut = now - timedelta(hours=hours)
    chunk = [s for ts, s in series if ts >= cut]
    return float(mean(chunk)) if chunk else 0.0


def _classify_direction(slope: float, r2: float) -> str:
    if r2 < 0.15 or abs(slope) < 0.005:
        return "flat"
    return "rising" if slope > 0 else "falling"


def compute_trend(posts: Iterable, *, team: str, bucket_hours: int = 6,
                  now: Optional[datetime] = None) -> TrendReport:
    series = _to_score_series(posts, team=team)
    if not series:
        return TrendReport()

    buckets = _bucketize(series, bucket_hours)
    slope, r2 = _ols_slope(buckets)
    volatility = pstdev([b.mean_score for b in buckets]) if len(buckets) > 1 else 0.0
    anomalies = _detect_anomalies(buckets)
    direction = _classify_direction(slope, r2)

    # Windowed momentum: last X hours vs preceding X hours
    now = now or datetime.now(timezone.utc)

    def window(hours: int) -> float:
        recent = _window_mean(series, hours, now=now)
        prior = [s for ts, s in series
                 if (now - timedelta(hours=2 * hours)) <= ts < (now - timedelta(hours=hours))]
        prior_mean = float(mean(prior)) if prior else 0.0
        return recent - prior_mean

    momentum_24h = window(24)
    momentum_6h = window(6)

    slope_strength = float(min(1.0, abs(slope)) * r2)

    return TrendReport(
        slope_per_hour=slope,
        slope_strength=slope_strength,
        r_squared=r2,
        volatility=float(volatility),
        momentum_24h=momentum_24h,
        momentum_6h=momentum_6h,
        direction=direction,
        anomalies=anomalies,
        buckets=buckets,
    )
