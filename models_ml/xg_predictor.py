"""EXTEND-05: optional XGBoost/LightGBM xG regressor.

Loaded lazily from `models_ml/artifacts/xg_predictor.json`. When the artifact
or library is missing, callers MUST fall back to the heuristic predictor in
`analysis.match_predictor` — see `match_service` for the wiring.

Training scaffold (`train_xg_predictor.py`) lives next to this file. It is
called manually when historical match data with xG labels becomes
available; the resulting JSON is committed to artifacts/.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger("models_ml.xg_predictor")

_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "xg_predictor.json"
_FEATURE_ORDER = [
    "elo_delta",
    "home_avg_xg",
    "away_avg_xg",
    "home_avg_xg_conceded",
    "away_avg_xg_conceded",
    "home_form_pts",
    "away_form_pts",
    "home_sentiment",
    "away_sentiment",
    "home_momentum",
    "away_momentum",
    "home_advantage",
    "venue_altitude_m",
    "rest_delta",
    "h2h_score",
]


@dataclass
class XgPrediction:
    home_xg: float
    away_xg: float


class XgPredictor:
    """Tiny linear fallback: if no real model artifact exists we use a hand-
    tuned coefficient set so the integration can still be exercised. As soon
    as a real LightGBM model lands, this class loads it instead.
    """

    def __init__(self) -> None:
        self._artifact: Optional[dict] = None
        self._available = _ARTIFACT_PATH.exists()
        self._tried_load = False

    def _load(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        if not _ARTIFACT_PATH.exists():
            self._available = False
            return
        try:
            self._artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("xg_artifact_load_failed", error=str(exc))
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(self, features: dict[str, float]) -> Optional[XgPrediction]:
        self._load()
        if not self._available or not self._artifact:
            return None
        coeffs_h = self._artifact.get("home_coeffs", {})
        coeffs_a = self._artifact.get("away_coeffs", {})
        intercept_h = float(self._artifact.get("home_intercept", 1.2))
        intercept_a = float(self._artifact.get("away_intercept", 1.2))

        def dot(coeffs: dict, intercept: float) -> float:
            x = intercept
            for k in _FEATURE_ORDER:
                x += float(coeffs.get(k, 0.0)) * float(features.get(k, 0.0))
            # Soft-positive transform — xG cannot be negative.
            return max(0.15, math.log1p(math.exp(min(x, 6.0))))

        return XgPrediction(home_xg=dot(coeffs_h, intercept_h), away_xg=dot(coeffs_a, intercept_a))


predictor = XgPredictor()
