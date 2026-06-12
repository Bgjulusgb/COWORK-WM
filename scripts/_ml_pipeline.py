"""Wiederverwendbare ML-Pipeline-Snippets.

Die hier definierten Funktionen liefern Code-Strings, die ueber
``python -c <code>`` als Subprozess ausgefuehrt werden. Das ist dieselbe
Strategie wie zuvor in ``menu.py``, nur dass die Strings jetzt zentral
liegen und sowohl vom interaktiven Menue als auch von der headless
Train-Pipeline (``ki_runner.train_pipeline``) genutzt werden.
"""
from __future__ import annotations


def inline_optuna(trials: int = 100) -> str:
    return (
        "from analysis.weight_optimizer import tune_weights, synthetic_brier_objective; "
        f"r = tune_weights(synthetic_brier_objective(targets=[1.0]), n_trials={trials}); "
        "print(f'\\nBester Brier-Score: {r.best_value:.6f}'); "
        "print(f'Artefakt: {r.artifact_path}'); "
        "print('Beste Gewichte:'); "
        "[print(f'  {k}: {v:.4f}') for k, v in sorted(r.best_params.items())]"
    )


def inline_pymc(draws: int = 2000, tune: int = 1000) -> str:
    return (
        "from analysis.bayes_weights import fit_posterior; "
        "from analysis.weight_optimizer import synthetic_brier_objective; "
        "obj = synthetic_brier_objective(targets=[1.0]); "
        f"r = fit_posterior(lambda p: -obj(p), draws={draws}, tune={tune}); "
        "print('\\nPosterior-Mittelwerte:'); "
        "[print(f'  {k}: {v:.4f}  (CI: {r.ci_low[k]:.4f} - {r.ci_high[k]:.4f}, R-hat: {r.r_hat[k]:.3f})') "
        "for k, v in sorted(r.mean.items())]"
    )


def inline_pagerank() -> str:
    return (
        "from pathlib import Path; "
        "from analysis.network_strength import build_network_pagerank; "
        "from scripts.team_real_data import TEAM_REAL_DATA; "
        "from datetime import datetime, timezone; "
        "matches = []; "
        "fifa = {c: d.get('world_ranking', 50) for c, d in TEAM_REAL_DATA.items()}; "
        "scores = build_network_pagerank(matches if matches else "
        "[{'home': c, 'away': 'XXX', 'home_goals': 1, 'away_goals': 0, "
        "'kickoff': datetime(2025,1,1,tzinfo=timezone.utc)} for c in list(fifa)[:10]], "
        "fifa_rank=fifa); "
        "print(f'\\n{len(scores)} Teams bewertet.'); "
        "top = sorted(scores.items(), key=lambda x: -x[1])[:10]; "
        "print('Top 10:'); "
        "[print(f'  {i+1:>2}. {c}: {s:.6f}') for i, (c, s) in enumerate(top)]"
    )
