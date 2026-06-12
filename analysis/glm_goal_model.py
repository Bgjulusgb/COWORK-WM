"""Statsmodels GLM Poisson goal model.

Alternative to Dixon-Coles. Fits

    log(λ_team) = α_team(attack) - δ_opp(defence) + γ · home_indicator

via :class:`statsmodels.genmod.generalized_linear_model.GLM` with a Poisson
family (log link). Selected when ``settings.goal_model == "glm_poisson"``.

Used in match_predictor for inference: given fitted parameters and the v3
factor-tilt λ multipliers, return the same per-team rates :func:`build_goal_model`
already produces. Trained offline by ``scripts/train_glm_goal_model.py``.

Pandas + statsmodels are *required* for this path; the import is lazy so the
default Poisson/Dixon-Coles paths keep working without them installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from config.settings import settings

log = structlog.get_logger("analysis.glm_goal_model")

_ARTIFACT = settings.base_dir / "models_ml" / "artifacts" / "glm_goal_model.json"


@dataclass
class GlmGoalModel:
    """Lean inference-side wrapper. ``team_attack``/``team_defence`` come from
    the offline-fit artifact; ``home_advantage`` is the γ coefficient."""
    team_attack: dict[str, float]
    team_defence: dict[str, float]
    home_advantage: float
    intercept: float = 0.0

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        atk_h = self.team_attack.get(home.upper(), 0.0)
        def_h = self.team_defence.get(home.upper(), 0.0)
        atk_a = self.team_attack.get(away.upper(), 0.0)
        def_a = self.team_defence.get(away.upper(), 0.0)
        # log(λ_home) = intercept + atk_h - def_a + γ
        # log(λ_away) = intercept + atk_a - def_h
        import math

        lam_h = math.exp(self.intercept + atk_h - def_a + self.home_advantage)
        lam_a = math.exp(self.intercept + atk_a - def_h)
        return lam_h, lam_a


def load_glm_artifact(path: Path | None = None) -> GlmGoalModel | None:
    p = path or _ARTIFACT
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("glm_artifact_load_failed", error=str(exc))
        return None
    return GlmGoalModel(
        team_attack=data.get("team_attack", {}),
        team_defence=data.get("team_defence", {}),
        home_advantage=float(data.get("home_advantage", 0.25)),
        intercept=float(data.get("intercept", 0.0)),
    )


def fit_glm_from_matches(matches: list[dict]) -> GlmGoalModel:
    """Fit on ``[{"home":CODE, "away":CODE, "home_goals":int, "away_goals":int}, ...]``.

    Builds the long format DataFrame statsmodels expects and writes the
    coefficients back to the artifact path. Returns the fitted model.
    """
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
    except Exception as exc:
        raise RuntimeError(
            "glm_goal_model needs pandas + statsmodels. "
            "pip install pandas>=2.2 statsmodels>=0.14"
        ) from exc

    rows: list[dict] = []
    for m in matches:
        h = m["home"].upper()
        a = m["away"].upper()
        rows.append({"team": h, "opp": a, "is_home": 1, "goals": int(m["home_goals"])})
        rows.append({"team": a, "opp": h, "is_home": 0, "goals": int(m["away_goals"])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no matches to fit on")

    team_dummies = pd.get_dummies(df["team"], prefix="atk", drop_first=True)
    opp_dummies = pd.get_dummies(df["opp"], prefix="def", drop_first=True)
    X = pd.concat([team_dummies, opp_dummies, df[["is_home"]]], axis=1).astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = df["goals"].astype(float).values

    model = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=200)
    params = model.params.to_dict()

    team_attack: dict[str, float] = {}
    team_defence: dict[str, float] = {}
    home_advantage = float(params.get("is_home", 0.25))
    intercept = float(params.get("const", 0.0))
    for key, val in params.items():
        if key.startswith("atk_"):
            team_attack[key.split("_", 1)[1]] = float(val)
        elif key.startswith("def_"):
            team_defence[key.split("_", 1)[1]] = float(val)

    artifact = {
        "team_attack": team_attack,
        "team_defence": team_defence,
        "home_advantage": home_advantage,
        "intercept": intercept,
    }
    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return GlmGoalModel(
        team_attack=team_attack,
        team_defence=team_defence,
        home_advantage=home_advantage,
        intercept=intercept,
    )


__all__ = ["GlmGoalModel", "load_glm_artifact", "fit_glm_from_matches"]
