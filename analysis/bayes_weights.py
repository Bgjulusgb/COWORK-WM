"""PyMC Bayesian posterior over factor weights — offline tool.

Models each factor weight as a TruncatedNormal centred on its current default
with a relatively flat prior. The likelihood is the Brier score of the
ensemble's per-match outcome distribution on the historical corpus.

Result artefact: ``models_ml/artifacts/bayes_weights.json`` with mean + 90 %
credible interval per weight. Used as:

    * A more honest "where should this weight be" answer than Optuna's point
      estimate (Optuna ignores posterior width).
    * A way to *prune* weights whose 90 % CI brackets zero — those factors
      can safely run at weight 0.

Because MCMC is expensive (minutes on a normal corpus), this is **not** a
runtime dependency. The Admin-Panel can read the JSON and surface the CI bars,
but predictions themselves use the runtime-loaded point estimates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import structlog

from analysis.weight_optimizer import _PRIOR_RANGES
from config.settings import settings

log = structlog.get_logger("analysis.bayes_weights")

_ARTIFACT = settings.base_dir / "models_ml" / "artifacts" / "bayes_weights.json"


@dataclass
class WeightPosterior:
    mean: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    r_hat: dict[str, float]


def fit_posterior(
    likelihood_fn: Callable[[dict[str, float]], float],
    *,
    draws: int = 2000,
    tune: int = 1000,
    keys: Sequence[str] | None = None,
) -> WeightPosterior:
    """Sample the posterior using PyMC's NUTS.

    ``likelihood_fn`` returns the **log-likelihood** of the corpus under the
    given weight vector. We turn that into a Potential so the user can plug
    any backtesting harness in.
    """
    try:
        import arviz as az
        import numpy as np
        import pymc as pm
    except Exception as exc:
        raise RuntimeError(
            "bayes_weights needs pymc + arviz. "
            "pip install pymc>=5.16.0 arviz>=0.18"
        ) from exc

    space = list(keys or _PRIOR_RANGES.keys())
    with pm.Model() as model:
        ws = {}
        for k in space:
            lo, hi = _PRIOR_RANGES[k]
            mu = (lo + hi) / 2
            sd = (hi - lo) / 4
            ws[k] = pm.TruncatedNormal(k, mu=mu, sigma=sd, lower=lo, upper=hi)

        def _logp(*args):
            params = {k: float(v) for k, v in zip(space, args)}
            return likelihood_fn(params)

        pm.Potential("ll", _logp(*[ws[k] for k in space]))
        trace = pm.sample(
            draws=draws, tune=tune,
            chains=2, cores=1, progressbar=False,
            target_accept=0.9, random_seed=42,
        )

    summary = az.summary(trace, hdi_prob=0.9)
    mean = {k: float(summary.loc[k, "mean"]) for k in space}
    ci_low = {k: float(summary.loc[k, "hdi_5%"]) for k in space}
    ci_high = {k: float(summary.loc[k, "hdi_95%"]) for k in space}
    r_hat = {k: float(summary.loc[k, "r_hat"]) for k in space}

    artifact = {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r_hat": r_hat,
    }
    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log.info("posterior_persisted", path=str(_ARTIFACT), n=draws)
    return WeightPosterior(mean=mean, ci_low=ci_low, ci_high=ci_high, r_hat=r_hat)


def load_posterior() -> WeightPosterior | None:
    try:
        data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return WeightPosterior(
        mean=data.get("mean", {}),
        ci_low=data.get("ci_low", {}),
        ci_high=data.get("ci_high", {}),
        r_hat=data.get("r_hat", {}),
    )


__all__ = ["fit_posterior", "load_posterior", "WeightPosterior"]
