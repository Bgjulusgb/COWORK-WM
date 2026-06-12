#!/usr/bin/env python3
"""Offline tuning of the goal-model parameters (blend weights + Dixon-Coles ρ) to
the **Ranked Probability Score** — the proper, order-aware 1X2 metric.

Reads a CSV of historical matches with their model λ inputs::

    home_xg,away_xg,home_score,away_score
    1.9,0.8,2,0
    1.1,1.4,1,2
    ...

For each Optuna trial it rebuilds the 3-model blend at the candidate ρ + blend
weights, recomputes the 1X2 for every row, and minimises the mean RPS. Writes the
best params to ``models_ml/artifacts/tuned_model_params.yaml`` — inspect them and
set ``settings.dixon_coles_rho`` (+ the blend weights) accordingly.

Optional dependency: needs ``optuna`` (``pip install -e .[tune]``).

    python scripts/tune_models_offline.py history.csv --trials 200
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = ("home_xg", "away_xg", "home_score", "away_score")


def rows_from_csv(path: str | Path) -> list[SimpleNamespace]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tuning CSV not found: {p}")
    rows: list[SimpleNamespace] = []
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        for raw in reader:
            try:
                if raw["home_score"] in (None, "") or raw["away_score"] in (None, ""):
                    continue
                rows.append(SimpleNamespace(
                    home_xg=float(raw["home_xg"]), away_xg=float(raw["away_xg"]),
                    actual_home_score=int(float(raw["home_score"])),
                    actual_away_score=int(float(raw["away_score"]))))
            except (TypeError, ValueError):
                continue
    return rows


def _predict_fn(params, row):
    from models_ml.poisson_goals import blend_score_matrix, build_all_goal_models
    from analysis.weight_optimizer import normalise_blend
    from wm2026.markets import one_x_two
    rho = float(params.get("dixon_coles_rho", 0.1))
    models = build_all_goal_models(rho=rho)
    weights = normalise_blend(params) or None
    p = one_x_two(blend_score_matrix(models, row.home_xg, row.away_xg, weights))
    return (p["home"], p["draw"], p["away"])


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    trials = 200
    if "--trials" in args:
        trials = int(args[args.index("--trials") + 1])

    rows = rows_from_csv(args[0])
    if len(rows) < 10:
        print(f"need ≥10 matches, got {len(rows)}", file=sys.stderr)
        return 1

    from analysis.weight_optimizer import (
        rps_objective_from_results, tune_model_params,
    )
    keys = ["blend_poisson", "blend_negbin", "blend_glm_poisson", "dixon_coles_rho"]
    obj = rps_objective_from_results(rows, predict_fn=_predict_fn)
    baseline = obj({"blend_poisson": 0.4, "blend_negbin": 0.3, "blend_glm_poisson": 0.3,
                    "dixon_coles_rho": 0.1})
    try:
        res = tune_model_params(obj, n_trials=trials, keys=keys)
    except RuntimeError as exc:
        print(f"{exc}\nInstall it with:  pip install -e .[tune]", file=sys.stderr)
        return 1
    print(f"tuned on {len(rows)} matches · RPS {baseline:.4f} (default) → {res.best_value:.4f} (tuned)")
    print("best params:")
    for k, v in sorted(res.best_params.items()):
        print(f"  {k}: {v:.4f}")
    print(f"→ wrote {res.artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
