"""Optuna-driven Bayesian search over the factor-weight space.

Goal: minimise the weighted Brier score on historical match results. Each
trial samples ``factor_weight_*`` values from prior ranges, applies them via
:meth:`Settings.reload_runtime_weights`, then runs ``backtesting`` over a
hold-out fold.

This is the most data-driven way to set the weights — and the result is the
seed for the deeper Bayesian posterior in ``bayes_weights`` (PyMC).

Optuna and the historical match corpus are optional dependencies: importing
this module without them is fine, only :func:`tune_weights` raises.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import structlog

from config.settings import settings

log = structlog.get_logger("analysis.weight_optimizer")

# Prior ranges per factor — keep modest so the result is still interpretable.
_PRIOR_RANGES: dict[str, tuple[float, float]] = {
    "factor_weight_elo":            (0.10, 0.45),
    "factor_weight_form":           (0.05, 0.30),
    "factor_weight_h2h":            (0.05, 0.25),
    "factor_weight_goals":          (0.05, 0.25),
    "factor_weight_context":        (0.05, 0.20),
    "factor_weight_sentiment":      (0.00, 0.20),
    "factor_weight_squad":          (0.00, 0.15),
    "factor_weight_fifa_rank":      (0.00, 0.15),
    "factor_weight_rest_travel":    (0.00, 0.15),
    "factor_weight_altitude":       (0.00, 0.10),
    "factor_weight_market":         (0.00, 0.20),
    "factor_weight_weather":        (0.00, 0.10),
    "factor_weight_injury":         (0.00, 0.15),
    "factor_weight_momentum":       (0.00, 0.15),
    "factor_weight_llm_sentiment":  (0.00, 0.15),
    "factor_weight_lineup":         (0.00, 0.10),
    "factor_weight_squad_value":    (0.00, 0.10),
    "factor_weight_network":        (0.00, 0.10),
    "factor_weight_ml":             (0.00, 0.15),
    "factor_weight_ml_lgbm":        (0.00, 0.10),
}

_ARTIFACT_PATH = settings.base_dir / "models_ml" / "artifacts" / "tuned_weights.yaml"


@dataclass
class TuningResult:
    best_value: float
    best_params: dict[str, float]
    n_trials: int
    artifact_path: Path


def tune_weights(
    objective: Callable[[dict[str, float]], float],
    *,
    n_trials: int = 100,
    study_name: str = "factor_weights",
    keys: Sequence[str] | None = None,
    storage: str | None = None,
) -> TuningResult:
    """Run an Optuna study against ``objective``.

    The user passes the objective so the optimizer can be exercised against
    either the live backtester or a small synthetic harness in tests.
    """
    try:
        import optuna
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "weight_optimizer requires optuna + pyyaml. "
            "pip install optuna>=3.6.0 PyYAML"
        ) from exc

    space = list(keys or _PRIOR_RANGES.keys())

    def _objective(trial: "optuna.trial.Trial") -> float:
        params = {
            k: trial.suggest_float(k, *_PRIOR_RANGES[k]) for k in space if k in _PRIOR_RANGES
        }
        return objective(params)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    study = optuna.create_study(
        direction="minimize", sampler=sampler,
        study_name=study_name, storage=storage,
        load_if_exists=True,
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_PATH.write_text(
        yaml.safe_dump(study.best_params, sort_keys=True), encoding="utf-8"
    )
    # Also drop the same payload as the live "runtime_weights" loader sees so
    # the Admin-Panel can hot-load the tuning result without a restart.
    runtime_path = settings.base_dir / "models_ml" / "artifacts" / "runtime_weights.yaml"
    runtime_path.write_text(
        yaml.safe_dump(study.best_params, sort_keys=True), encoding="utf-8"
    )
    log.info("weights_tuned", best_value=study.best_value, n_trials=n_trials)
    return TuningResult(
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=n_trials,
        artifact_path=_ARTIFACT_PATH,
    )


# ── Phase 4 — model-parameter tuning to RPS (blend weights, ρ, ξ) ─────────────
_MODEL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "blend_poisson":      (0.20, 0.60),
    "blend_negbin":       (0.10, 0.50),
    "blend_glm_poisson":  (0.10, 0.50),
    "dixon_coles_rho":    (0.00, 0.18),
    "mle_time_decay_xi":  (0.001, 0.02),
}
_MODEL_ARTIFACT_PATH = settings.base_dir / "models_ml" / "artifacts" / "tuned_model_params.yaml"


def tune_model_params(
    objective: Callable[[dict[str, float]], float],
    *,
    n_trials: int = 100,
    keys: Sequence[str] | None = None,
    study_name: str = "model_params",
) -> TuningResult:
    """Optuna study over the goal-model params (blend weights / ρ / ξ), minimising
    ``objective`` (typically mean RPS over a backtest). Writes the best params to
    ``models_ml/artifacts/tuned_model_params.yaml`` for inspection / hot-load.
    Optional dep: raises if optuna is missing."""
    try:
        import optuna
        import yaml
    except Exception as exc:
        raise RuntimeError("tune_model_params requires optuna + pyyaml") from exc

    space = list(keys or _MODEL_PARAM_RANGES.keys())

    def _objective(trial: "optuna.trial.Trial") -> float:
        params = {k: trial.suggest_float(k, *_MODEL_PARAM_RANGES[k])
                  for k in space if k in _MODEL_PARAM_RANGES}
        return objective(params)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    _MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MODEL_ARTIFACT_PATH.write_text(yaml.safe_dump(study.best_params, sort_keys=True), encoding="utf-8")
    log.info("model_params_tuned", best_value=study.best_value, n_trials=n_trials)
    return TuningResult(float(study.best_value), dict(study.best_params), n_trials, _MODEL_ARTIFACT_PATH)


def normalise_blend(params: dict[str, float]) -> dict[str, float]:
    """Renormalise the three sampled blend weights to sum 1 (for blend_score_matrix)."""
    keys = ("blend_poisson", "blend_negbin", "blend_glm_poisson")
    present = {k: max(0.0, params[k]) for k in keys if k in params}
    total = sum(present.values()) or 1.0
    return {("poisson" if k == "blend_poisson" else
             "negbin" if k == "blend_negbin" else "glm_poisson"): v / total
            for k, v in present.items()}


def rps_objective_from_results(rows, *, predict_fn: Callable[[dict, object], tuple]) -> Callable[[dict], float]:
    """Build an RPS objective over historical ``rows``. ``predict_fn(params, row)``
    returns a 1X2 triple (home, draw, away) under the candidate params; the
    objective returns the mean Ranked Probability Score (lower = better)."""
    from analysis.backtesting import _outcome_vec, _rps

    def obj(params: dict[str, float]) -> float:
        total = 0.0
        n = 0
        for r in rows:
            if getattr(r, "actual_home_score", None) is None or getattr(r, "actual_away_score", None) is None:
                continue
            p = predict_fn(params, r)
            y = _outcome_vec(int(r.actual_home_score), int(r.actual_away_score))
            total += _rps(tuple(p), y)
            n += 1
        return total / n if n else 1.0

    return obj


def synthetic_rps_objective(targets: list[float]) -> Callable[[dict[str, float]], float]:
    """Deterministic quadratic bowl with its minimum at the ``_MODEL_PARAM_RANGES``
    centres — exercises ``tune_model_params`` in tests without a match corpus."""
    centres = {k: (lo + hi) / 2 for k, (lo, hi) in _MODEL_PARAM_RANGES.items()}

    def obj(params: dict[str, float]) -> float:
        return sum((v - centres[k]) ** 2 for k, v in params.items()) + sum(targets) * 1e-6

    return obj


def synthetic_brier_objective(targets: list[float]) -> Callable[[dict[str, float]], float]:
    """Return an objective that maps each weight set to a deterministic Brier-
    like value — used by the unit tests so the optimizer can be exercised
    without the live match corpus.
    """
    centres = {k: (lo + hi) / 2 for k, (lo, hi) in _PRIOR_RANGES.items()}

    def obj(params: dict[str, float]) -> float:
        sse = 0.0
        for k, v in params.items():
            sse += (v - centres[k]) ** 2
        # Tie a tiny term to ``targets`` so the function still depends on
        # the caller's contract; smaller is better.
        return sse + sum(targets) * 1e-6

    return obj


__all__ = [
    "tune_weights", "TuningResult", "synthetic_brier_objective", "_PRIOR_RANGES",
    "tune_model_params", "rps_objective_from_results", "synthetic_rps_objective",
    "normalise_blend", "_MODEL_PARAM_RANGES",
]
