"""LightGBM blend factor — zweiter ML-Kopf neben XGBoost (factors/ml_blend.py).

Spiegelt MlBlendFactor, aber laedt den LightGBM-Booster aus
``models_ml/artifacts/xg_predictor_lgbm.txt``. Bei fehlendem Artifact
self-disable -> Factor-Ensemble re-normalisiert ihn raus. Aktivierung:
``factor_weight_ml_lgbm > 0`` UND ``scripts/train_xg_predictor.py --model lgbm``
gelaufen.

Symmetrische Vorhersage: derselbe Booster wird zweimal aufgerufen — einmal mit
"home perspective" (home_features = h, away_features = a), einmal getauscht.
Das passt zum Single-Output-Trainer in ``analysis/lightgbm_blend.py:train_lgbm``.
"""
from __future__ import annotations

import math

from analysis.lightgbm_blend import load_lgbm_artifact
from factors.base import Factor, FactorContext, FactorSignal
from factors.ml_blend import _features as _ml_features

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


def _swap_perspective(features: dict[str, float]) -> dict[str, float]:
    """Tausche home/away-Felder, kippe Differenz-Features."""
    out = dict(features)
    swaps = [
        ("home_avg_xg", "away_avg_xg"),
        ("home_avg_xg_conceded", "away_avg_xg_conceded"),
        ("home_form_pts", "away_form_pts"),
        ("home_sentiment", "away_sentiment"),
        ("home_momentum", "away_momentum"),
    ]
    for a, b in swaps:
        out[a], out[b] = features.get(b, 0.0), features.get(a, 0.0)
    out["elo_delta"] = -features.get("elo_delta", 0.0)
    out["rest_delta"] = -features.get("rest_delta", 0.0)
    out["h2h_score"] = -features.get("h2h_score", 0.0)
    out["home_advantage"] = 0.0
    return out


class MlBlendLgbmFactor(Factor):
    name = "ml_blend_lgbm"
    default_weight = 0.00

    def __init__(self, weight: float = 0.00) -> None:
        super().__init__(weight)
        self._artifact = None
        self._tried_load = False

    def _ensure_loaded(self):
        if self._tried_load:
            return self._artifact
        self._tried_load = True
        self._artifact = load_lgbm_artifact()
        return self._artifact

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        artifact = self._ensure_loaded()
        if artifact is None:
            return self._neutral(source="ml", reason="no_lgbm_artifact")

        try:
            import numpy as np
        except Exception:
            return self._neutral(source="ml", reason="numpy_missing")

        feats = _ml_features(ctx)
        feat_names = artifact.feature_names or _FEATURE_ORDER
        try:
            home_vec = np.array([[feats.get(k, 0.0) for k in feat_names]], dtype=float)
            away_vec = np.array(
                [[_swap_perspective(feats).get(k, 0.0) for k in feat_names]],
                dtype=float,
            )
            home_xg = float(artifact.booster.predict(home_vec)[0])
            away_xg = float(artifact.booster.predict(away_vec)[0])
        except Exception:
            return self._neutral(source="ml", reason="lgbm_predict_failed")

        home_xg = max(0.15, home_xg)
        away_xg = max(0.15, away_xg)
        if home_xg <= 0 or away_xg <= 0:
            return self._neutral(source="ml", reason="lgbm_zero_output")

        g = math.sqrt(home_xg * away_xg)
        home_strength = max(0.5, min(2.0, home_xg / g))
        away_strength = max(0.5, min(2.0, away_xg / g))

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=0.6,
            available=True,
            source="lgbm_model",
            raw_data={
                "model_home_xg": round(home_xg, 3),
                "model_away_xg": round(away_xg, 3),
                "is_xg_proxy": True,
            },
        )
