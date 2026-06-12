"""Phase 5.1 — Robustheits-Check gegen Input-Bias (λ-Perturbation).

Der Bootstrap-p5 in `wm2026.edge` misst **Sampling-Rauschen** bei festen λ.
Was er nicht misst: den Bias der λ-treibenden Inputs selbst (illustrative
YAML-xG, geschätztes Elo). Dieser Check perturbiert die finalen Blend-λ um
±`rel` (Default 15 %) auf einem 3×3-Grid und bewertet jede bepreiste Selektion
in jedem Szenario neu — direkt auf der Blend-Matrix, ohne Pipeline-Re-Run.

`robust_pct` = Anteil der Szenarien, in denen die Edge positiv bleibt. Ein
Pick, dessen Edge bei ±15 % λ-Unsicherheit das Vorzeichen wechselt, ist kein
Value, sondern ein Datenartefakt (vgl. CAN–BIH 2026-06-12: p5 +34 % → −18 %
unter Tor-Proxy-xG).

Alles hier ist eine **lineare Funktion der Score-Matrix** — wir nutzen
`blend_score_matrix` (Konsistenz-Contract aus CLAUDE.md) und reine
numpy-Summen. Keine neuen Dependencies.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["sensitivity_check"]

# Selektionen, die wir aus der Matrix bewerten können → Funktional.
# M[i][j] = P(home i, away j).
def _p_home(M: np.ndarray) -> float:
    return float(np.tril(M, -1).sum())          # i > j


def _p_draw(M: np.ndarray) -> float:
    return float(np.trace(M))


def _p_away(M: np.ndarray) -> float:
    return float(np.triu(M, 1).sum())           # j > i


def _p_over25(M: np.ndarray) -> float:
    n = M.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(M.shape[1]))
    return float(M[idx >= 3].sum())


def _p_btts(M: np.ndarray) -> float:
    return float(M[1:, 1:].sum())


_FUNCTIONALS = {
    ("1X2", "Home"): _p_home,
    ("1X2", "Draw"): _p_draw,
    ("1X2", "Away"): _p_away,
    ("O/U 2.5", "Over 2.5"): _p_over25,
    ("O/U 2.5", "Under 2.5"): lambda M: 1.0 - _p_over25(M),
    ("BTTS", "Yes"): _p_btts,
    ("BTTS", "No"): lambda M: 1.0 - _p_btts(M),
}


def sensitivity_check(
    models: Sequence[Any],
    lam_home: float,
    lam_away: float,
    edge_rows: Sequence[dict[str, Any]],
    *,
    rel: float = 0.15,
    steps: Sequence[float] = (-1.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Bewerte jede bepreiste Edge-Zeile unter λ-Perturbation.

    Returns (additiv ins Report-JSON als ``sensitivity``)::

        {"rel": 0.15, "n_scenarios": 9,
         "selections": {"1X2/Away": {"robust_pct": 0.44,
                                      "edge_min_pct": -18.2, "edge_max_pct": 31.0}},
         "fragile": ["1X2/Away"]}    # Edges, die das Vorzeichen wechseln
    """
    from models_ml.poisson_goals import blend_score_matrix

    rows = [r for r in edge_rows
            if r.get("decimal_odd") and (r["market"], r["selection"]) in _FUNCTIONALS]
    out: dict[str, Any] = {"rel": rel, "n_scenarios": len(steps) ** 2,
                           "selections": {}, "fragile": []}
    if not rows:
        return out

    # Matrizen für alle Szenarien einmal bauen (9 Blend-Matrizen).
    scenarios = []
    for dh in steps:
        for da in steps:
            lh = max(0.05, lam_home * (1.0 + rel * dh))
            la = max(0.05, lam_away * (1.0 + rel * da))
            scenarios.append(np.asarray(blend_score_matrix(models, lh, la)))

    for r in rows:
        fn = _FUNCTIONALS[(r["market"], r["selection"])]
        odd = float(r["decimal_odd"])
        edges = [fn(M) * odd - 1.0 for M in scenarios]
        key = f"{r['market']}/{r['selection']}"
        robust = sum(1 for e in edges if e > 0) / len(edges)
        out["selections"][key] = {
            "robust_pct": round(robust, 4),
            "edge_min_pct": round(100 * min(edges), 2),
            "edge_max_pct": round(100 * max(edges), 2),
        }
        # fragil = Vorzeichenwechsel über die Szenarien (Edge nicht robust)
        if min(edges) < 0 < max(edges):
            out["fragile"].append(key)

    return out
