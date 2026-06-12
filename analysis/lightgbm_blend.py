"""LightGBM blend factor — second ML head alongside XGBoost.

The XGBoost path lives in ``factors/ml_blend.py`` + ``scripts/train_xg_predictor.py``.
We mirror it with LightGBM so the ensemble can hold *two* ML opinions whose
disagreement is itself informative.

Artifact: ``models_ml/artifacts/xg_predictor_lgbm.txt`` (LightGBM's native
model save format). ``factor_weight_ml_lgbm = 0`` (default) ⇒ off.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import structlog

from config.settings import settings

log = structlog.get_logger("analysis.lightgbm_blend")

_ARTIFACT_MODEL = settings.base_dir / "models_ml" / "artifacts" / "xg_predictor_lgbm.txt"
_ARTIFACT_META = settings.base_dir / "models_ml" / "artifacts" / "xg_predictor_lgbm.json"


@dataclass
class LgbmArtifact:
    booster: Any                      # lightgbm.Booster
    feature_names: list[str]
    target_mean: float
    target_std: float


def load_lgbm_artifact() -> LgbmArtifact | None:
    if not _ARTIFACT_MODEL.exists():
        return None
    try:
        import lightgbm as lgb
    except Exception:
        log.debug("lightgbm_missing")
        return None
    try:
        booster = lgb.Booster(model_file=str(_ARTIFACT_MODEL))
        meta = json.loads(_ARTIFACT_META.read_text(encoding="utf-8"))
        return LgbmArtifact(
            booster=booster,
            feature_names=list(meta.get("feature_names", [])),
            target_mean=float(meta.get("target_mean", 1.3)),
            target_std=float(meta.get("target_std", 0.5)),
        )
    except Exception as exc:
        log.warning("lgbm_artifact_load_failed", error=str(exc))
        return None


def train_lgbm(features: Sequence[dict], targets: Sequence[float]) -> Path:
    """Fit a LightGBM regressor on prepared feature dicts.

    Walk-forward CV happens in the caller (``scripts/train_xg_predictor_lgbm.py``);
    this function just does a single deterministic fit and writes both
    artifacts so the factor can hot-reload.
    """
    try:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "lightgbm_blend.train_lgbm needs lightgbm + pandas + numpy. "
            "pip install lightgbm>=4.5.0"
        ) from exc

    df = pd.DataFrame(features)
    y = np.asarray(targets, dtype=float)
    if len(df) != len(y):
        raise ValueError("feature/target length mismatch")
    train_set = lgb.Dataset(df, label=y)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 8,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
    }
    booster = lgb.train(params, train_set, num_boost_round=400)
    _ARTIFACT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(_ARTIFACT_MODEL))
    _ARTIFACT_META.write_text(json.dumps({
        "feature_names": list(df.columns),
        "target_mean": float(y.mean()),
        "target_std": float(y.std() or 1.0),
    }, indent=2), encoding="utf-8")
    return _ARTIFACT_MODEL


__all__ = ["LgbmArtifact", "load_lgbm_artifact", "train_lgbm"]
