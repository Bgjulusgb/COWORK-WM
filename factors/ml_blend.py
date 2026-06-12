"""ML-blend factor (EXTEND-05) — wraps the trained xG regressor.

Turns `models_ml/xg_predictor.predictor` into a normal Factor so a trained model
can nudge xG alongside the heuristic signals instead of replacing the pipeline.
When no artifact is trained (the default) `predictor.is_available` is False and
this factor returns a neutral, unavailable signal — so it costs nothing and the
ensemble re-normalises it out. Train via `scripts/train_xg_predictor.py`, then
set FACTOR_WEIGHT_ML > 0.

The model outputs (home_xg, away_xg); we map them to strengths with the same
geometric-mean normalisation GoalEfficiencyFactor uses (so home·away ≈ 1).
"""
from __future__ import annotations

import math

from factors._history import team_rows, weighted_goal_rates
from factors.base import Factor, FactorContext, FactorSignal

_FORM_PTS = {"W": 3, "D": 1, "L": 0}


def _form_pts(form) -> int:
    if not isinstance(form, list):
        return 7
    return sum(_FORM_PTS.get(str(c).upper(), 0) for c in form)


def _h2h_score(ctx: FactorContext) -> float:
    hw = aw = n = 0
    home = ctx.home_code.upper()
    for m in ctx.head_to_head or []:
        if getattr(m, "home_score", None) is None:
            continue
        mh = (getattr(m, "home_code", "") or "").upper()
        hs, as_ = int(m.home_score), int(m.away_score)
        home_goals, away_goals = (hs, as_) if mh == home else (as_, hs)
        if home_goals > away_goals:
            hw += 1
        elif home_goals < away_goals:
            aw += 1
        n += 1
    return (hw - aw) / n if n else 0.0


def _features(ctx: FactorContext) -> dict[str, float]:
    teams = (ctx.config or {}).get("teams") or {}
    h, a = teams.get("home") or {}, teams.get("away") or {}
    sp = ctx.sentiment_payload or {}
    alt = getattr(ctx.venue_info, "altitude_m", None) or 0.0
    rest_delta = 0.0
    if ctx.rest_days_home is not None and ctx.rest_days_away is not None:
        rest_delta = float(ctx.rest_days_home - ctx.rest_days_away)
    return {
        "elo_delta": float((h.get("elo_rating") or 1500) - (a.get("elo_rating") or 1500)),
        "home_avg_xg": float(h.get("avg_xg_season") or 1.3),
        "away_avg_xg": float(a.get("avg_xg_season") or 1.3),
        "home_avg_xg_conceded": float(h.get("avg_xg_conceded") or 1.3),
        "away_avg_xg_conceded": float(a.get("avg_xg_conceded") or 1.3),
        "home_form_pts": float(_form_pts(h.get("form_last5"))),
        "away_form_pts": float(_form_pts(a.get("form_last5"))),
        "home_sentiment": float(sp.get("home_sentiment", 0.0)),
        "away_sentiment": float(sp.get("away_sentiment", 0.0)),
        "home_momentum": float(sp.get("home_momentum", 0.0)),
        "away_momentum": float(sp.get("away_momentum", 0.0)),
        "home_advantage": 0.0,
        "venue_altitude_m": float(alt),
        "rest_delta": rest_delta,
        "h2h_score": _h2h_score(ctx),
    }


class MlBlendFactor(Factor):
    name = "ml_blend"
    default_weight = 0.00

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        try:
            from models_ml.xg_predictor import predictor
        except Exception:
            return self._neutral(source="ml", reason="predictor_import_failed")

        if not predictor.is_available:
            return self._neutral(source="ml", reason="no_trained_artifact")

        out = predictor.predict(_features(ctx))
        if out is None or out.home_xg <= 0 or out.away_xg <= 0:
            return self._neutral(source="ml", reason="model_returned_nothing")

        g = math.sqrt(out.home_xg * out.away_xg)
        home_strength = max(0.5, min(2.0, out.home_xg / g))
        away_strength = max(0.5, min(2.0, out.away_xg / g))

        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=0.65,
            available=True,
            source="ml_model",
            raw_data={"model_home_xg": round(out.home_xg, 3),
                      "model_away_xg": round(out.away_xg, 3), "is_xg_proxy": True},
        )
