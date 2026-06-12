"""EXTEND-07: sentiment-drift alarms.

Builds on the existing `trend_analyzer` anomaly detector. An alarm fires when
both conditions hold for the same time bucket:

  * |z_score| > _Z_THRESHOLD (default 3.0)
  * post_velocity > _VELOCITY_MULTIPLIER * baseline_velocity

In other words: the conversation suddenly got loud AND polarised. Either
condition alone is too noisy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


_Z_THRESHOLD = 3.0
_VELOCITY_MULTIPLIER = 2.0


@dataclass
class DriftAlert:
    team: str
    bucket_start: str
    z_score: float
    direction: str
    post_velocity: float
    baseline_velocity: float


def detect(
    anomalies: Iterable[dict],
    *,
    current_velocity: float,
    baseline_velocity: float,
    team: str,
) -> list[DriftAlert]:
    if baseline_velocity <= 0:
        return []
    if current_velocity < baseline_velocity * _VELOCITY_MULTIPLIER:
        return []
    out: list[DriftAlert] = []
    for a in anomalies:
        z = abs(float(a.get("z_score", 0.0)))
        if z >= _Z_THRESHOLD:
            out.append(
                DriftAlert(
                    team=team,
                    bucket_start=str(a.get("bucket_start", "")),
                    z_score=z,
                    direction=str(a.get("direction", "unknown")),
                    post_velocity=current_velocity,
                    baseline_velocity=baseline_velocity,
                )
            )
    return out
