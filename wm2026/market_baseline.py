"""Phase 6.2 — markt-implizite λ-Baseline (Inversion der vig-freien Quoten).

Das wiederkehrende Kernproblem des Workflows: Ohne öffentliche NT-xG ist die
YAML-Baseline illustrativ, und jede Modell-Markt-Divergenz ist dann Diagnose
der Datenlücke statt Value (siehe CAN–BIH 2026-06-12). Wenn aber recherchierte
Quoten vorliegen, trägt der Markt selbst die beste verfügbare λ-Information —
der kanonisch gut kalibrierte Forecaster (Constantinou & Fenton 2013).

``market_implied_lambdas`` invertiert deshalb die vig-freien 1X2- (und optional
O/U-2.5-) Wahrscheinlichkeiten zur (λ_home, λ_away), die das Blend-Modell auf
genau diese Marktlinie bringt: 2-Parameter-Least-Squares über der Blend-Matrix
(Nelder-Mead, log-Parametrisierung ⇒ λ > 0 garantiert).

Opt-in via ``wm2026 predict --baseline market`` — Default bleibt ``yaml``
(Output unverändert). Caveat (dokumentiert): kombiniert mit dem market_odds-
Faktor und ``--calibrate market`` fließt die Marktinfo dreifach ein; für reine
Markt-Konsistenz-Analysen gedacht, nicht für Edge-Jagd gegen denselben Markt.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

__all__ = ["market_implied_lambdas"]


def _probs(M: np.ndarray) -> tuple[float, float, float]:
    """(home, away, over2.5) als lineare Funktionale der Score-Matrix."""
    n = M.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(M.shape[1]))
    return (float(np.tril(M, -1).sum()),
            float(np.triu(M, 1).sum()),
            float(M[idx >= 3].sum()))


def market_implied_lambdas(
    models: Mapping[str, Any],
    fair_1x2: tuple[float, float, float],
    fair_over25: float | None = None,
    *,
    x0: tuple[float, float] = (1.35, 1.15),
    tol: float = 1e-10,
) -> tuple[float, float, Dict[str, float]]:
    """Finde (λ_home, λ_away), deren Blend-Matrix die Marktlinie reproduziert.

    Parameters
    ----------
    models       Blend-Modelle (wie ``MatchPredictor.models``).
    fair_1x2     vig-freie (home, draw, away) — z.B. aus ``wm2026.edge.devig``.
    fair_over25  optional vig-freie Over-2.5-Wahrscheinlichkeit; pinnt die
                 Tor-Summe (ohne sie ist die Gesamttor-Ebene nur schwach
                 durch die Draw-Wahrscheinlichkeit identifiziert).

    Returns ``(λ_home, λ_away, diag)`` mit Diagnose
    ``{"loss": …, "fit_home": …, "fit_away": …, "fit_over25": …}``.
    """
    from scipy.optimize import minimize

    from models_ml.poisson_goals import blend_score_matrix

    t_home, _t_draw, t_away = float(fair_1x2[0]), float(fair_1x2[1]), float(fair_1x2[2])

    def loss(x: np.ndarray) -> float:
        lh, la = float(np.exp(x[0])), float(np.exp(x[1]))
        M = blend_score_matrix(models, lh, la)
        ph, pa, po = _probs(np.asarray(M))
        err = (ph - t_home) ** 2 + (pa - t_away) ** 2
        if fair_over25 is not None:
            err += (po - float(fair_over25)) ** 2
        return err

    res = minimize(loss, x0=np.log(np.asarray(x0, dtype=float)),
                   method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": tol, "maxiter": 2000})
    lh, la = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    M = np.asarray(blend_score_matrix(models, lh, la))
    ph, pa, po = _probs(M)
    diag = {"loss": float(res.fun), "fit_home": round(ph, 4),
            "fit_away": round(pa, 4), "fit_over25": round(po, 4),
            "converged": bool(res.success)}
    return lh, la, diag
