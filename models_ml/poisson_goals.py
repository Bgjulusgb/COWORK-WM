from typing import Any, Dict, List, Tuple

import math

import numpy as np
from scipy.stats import nbinom, poisson


class DixonColesPoisson:
    """Dixon & Coles (1997) correction over independent Poisson goal counts.

    With ρ > 0, down-weights 0-0 and 1-1 (low-scoring draws are rarer than the
    independence assumption predicts) and up-weights 1-0/0-1 (narrow wins).
    Scores above (1,1) are left at the plain Poisson product.
    """

    def __init__(self, rho: float = 0.1, max_goals: int = 6) -> None:
        self.rho = rho
        self.max_goals = max_goals

    def _pmf(self, k: int, mu: float) -> float:
        """Marginal goal pmf for one side. Overridden by the negative-binomial
        model; Poisson is the Dixon-Coles default."""
        return float(poisson.pmf(k, mu))

    def _pmf_vector(self, n: int, mu: float) -> np.ndarray:
        """Vectorised marginal pmf for goals 0..n-1 — one scipy call instead of n
        scalar ``_pmf`` calls. Overridden by the negative-binomial model."""
        return poisson.pmf(np.arange(n), mu)

    def predict_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        # Independent-marginal outer product + the Dixon-Coles low-score
        # correction on the four affected cells. Mathematically identical to the
        # per-cell loop, but ~10-50× faster (one vectorised pmf per side) — which
        # the bootstrap (n× per match) and the tournament MC (per pairing) lean on.
        n = self.max_goals + 1
        matrix = np.outer(self._pmf_vector(n, home_xg), self._pmf_vector(n, away_xg))
        rho = self.rho
        matrix[0, 0] *= 1.0 - home_xg * away_xg * rho
        matrix[0, 1] *= 1.0 + home_xg * rho
        matrix[1, 0] *= 1.0 + away_xg * rho
        matrix[1, 1] *= 1.0 - rho
        np.clip(matrix, 0.0, None, out=matrix)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def _correction(self, h: int, a: int, mu: float, lam: float) -> float:
        rho = self.rho
        if h == 0 and a == 0:
            return 1 - mu * lam * rho
        if h == 0 and a == 1:
            return 1 + mu * rho
        if h == 1 and a == 0:
            return 1 + lam * rho
        if h == 1 and a == 1:
            return 1 - rho
        return 1.0

    def markets(self, matrix: np.ndarray) -> Dict[str, float | List[Dict]]:
        n = matrix.shape[0]
        home_win = float(np.sum(np.tril(matrix, -1)))
        draw = float(np.trace(matrix))
        away_win = float(np.sum(np.triu(matrix, 1)))

        def goals_over(threshold: float) -> float:
            return float(
                sum(matrix[i][j] for i in range(n) for j in range(n) if i + j > threshold)
            )

        btts = float(sum(matrix[i][j] for i in range(1, n) for j in range(1, n)))

        flat = [(float(matrix[i][j]), i, j) for i in range(n) for j in range(n)]
        flat.sort(reverse=True)
        top_scores = [
            {"home": h, "away": a, "probability": p} for p, h, a in flat[:5]
        ]

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_05": goals_over(0.5),
            "over_15": goals_over(1.5),
            "over_25": goals_over(2.5),
            "over_35": goals_over(3.5),
            "btts": btts,
            "top_scores": top_scores,
        }


class NegativeBinomialDixonColes(DixonColesPoisson):
    """Dixon-Coles over negative-binomial marginals instead of Poisson.

    Football goal counts are mildly over-dispersed (variance > mean): blowouts
    and 0-0s both occur more than a Poisson with the same mean predicts. The
    NB marginal adds a dispersion parameter `size` (r); as r → ∞ it converges
    back to Poisson. Mean is fixed to the supplied xG (μ) via p = r / (r + μ),
    so swapping the model keeps the expected scoreline while widening the tails.
    (Karlis & Ntzoufras 2003 discuss dispersion in football scorelines.)
    """

    def __init__(self, rho: float = 0.1, max_goals: int = 6, size: float = 8.0) -> None:
        super().__init__(rho=rho, max_goals=max_goals)
        # Guard: a tiny size makes the distribution wildly over-dispersed.
        self.size = max(1.0, float(size))

    def _pmf(self, k: int, mu: float) -> float:
        mu = max(1e-6, mu)
        r = self.size
        p = r / (r + mu)
        return float(nbinom.pmf(k, r, p))

    def _pmf_vector(self, n: int, mu: float) -> np.ndarray:
        mu = max(1e-6, mu)
        r = self.size
        return nbinom.pmf(np.arange(n), r, r / (r + mu))


class BivariatePoisson(DixonColesPoisson):
    """Karlis & Ntzoufras (2003) bivariate Poisson with a shared covariance term.

    Models positive goal correlation *globally* (not just on the four low scores
    the Dixon-Coles ρ touches) via a common component λ₃::

        X = Y₁ + Y₃,   Y = Y₂ + Y₃,   Yᵢ ~ Poisson(λᵢ),  Cov(X, Y) = λ₃ ≥ 0
        P(X=x, Y=y) = e^{-(λ₁+λ₂+λ₃)} · Σ_{k=0}^{min(x,y)}
                         λ₁^{x-k}/(x-k)! · λ₂^{y-k}/(y-k)! · λ₃^k/k!

    Crucially λ₃ is split *out of* the supplied xG (λ₁ = home_xg − λ₃,
    λ₂ = away_xg − λ₃), so the **marginal means stay exactly home_xg / away_xg** —
    the model adds correlation without moving the expected scoreline. λ₃ → 0
    recovers the independent Poisson product. The Dixon-Coles low-score
    correction is off by default here (ρ=0) since λ₃ is the correlation
    mechanism; ``markets()`` is inherited unchanged.
    """

    def __init__(self, lambda3: float = 0.12, max_goals: int = 6, rho: float = 0.0) -> None:
        super().__init__(rho=rho, max_goals=max_goals)
        self.lambda3 = max(0.0, float(lambda3))

    def predict_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        n = self.max_goals + 1
        # Keep λ₁, λ₂ ≥ 0: never pull more covariance than the smaller mean allows.
        l3 = max(0.0, min(self.lambda3, 0.9 * min(home_xg, away_xg)))
        l1 = max(1e-9, home_xg - l3)
        l2 = max(1e-9, away_xg - l3)
        base = math.exp(-(l1 + l2 + l3))
        matrix = np.zeros((n, n))
        for x in range(n):
            for y in range(n):
                s = 0.0
                for k in range(min(x, y) + 1):
                    s += (
                        l1 ** (x - k) / math.factorial(x - k)
                        * l2 ** (y - k) / math.factorial(y - k)
                        * l3 ** k / math.factorial(k)
                    )
                matrix[x][y] = base * s
        total = matrix.sum()
        return matrix / total if total > 0 else matrix


def build_goal_model(model: str = "poisson", *, rho: float = 0.1, max_goals: int = 6,
                     negbin_size: float = 8.0, lambda3: float = 0.12) -> DixonColesPoisson:
    """Factory used by MatchPredictor. `model` is "poisson" (default), "negbin",
    "glm_poisson" or "bivariate" (Karlis-Ntzoufras shared-covariance Poisson);
    anything else falls back to Poisson so a bad config never breaks a prediction.

    GLM-Variante: weniger Dixon-Coles-Korrektur (rho/2), weil der trainierte
    GLM bereits Heimvorteil + Team-Festeffekte modelliert. Praktischer Effekt
    auf gegebenes (home_xg, away_xg): leicht andere Low-Score-Gewichtung als
    die plain Poisson-Variante — so produziert das Modell-Ensemble bei
    gleichen Lambdas drei distinkte Vorhersagen.
    """
    m = (model or "poisson").lower()
    if m in ("negbin", "negative_binomial", "nb"):
        return NegativeBinomialDixonColes(rho=rho, max_goals=max_goals, size=negbin_size)
    if m in ("glm_poisson", "glm"):
        return DixonColesPoisson(rho=rho * 0.5, max_goals=max_goals)
    if m in ("bivariate", "bivariate_poisson", "bipois"):
        return BivariatePoisson(lambda3=lambda3, max_goals=max_goals)
    return DixonColesPoisson(rho=rho, max_goals=max_goals)


# ── Multi-Model Ensemble + Bootstrap ──────────────────────────────────────────

MODEL_NAMES: Tuple[str, ...] = ("poisson", "negbin", "glm_poisson")

DEFAULT_BLEND_WEIGHTS: Dict[str, float] = {
    "poisson": 0.40,
    "negbin": 0.30,
    "glm_poisson": 0.30,
}

# Phase 4 (Verbesserungsplan 2.2): Blend-Gewichte wenn das bivariate Poisson als
# 4. Modell mitlaeuft (settings.include_bivariate). Die Trio-Verhaeltnisse
# bleiben erhalten (0.34/0.25/0.25 ≈ 0.4/0.3/0.3 skaliert), λ₃-Korrelation
# bekommt einen bewusst moderaten Anteil bis ein Backtest-Tuning (2.4) vorliegt.
BLEND_WEIGHTS_WITH_BIVARIATE: Dict[str, float] = {
    "poisson": 0.34,
    "negbin": 0.25,
    "glm_poisson": 0.25,
    "bivariate": 0.16,
}


def resolve_blend_weights(model_names) -> Dict[str, float]:
    """Blend-Gewichte fuer eine konkrete Modell-Menge, renormalisiert auf 1.

    Enthaelt die Menge ``"bivariate"``, gilt :data:`BLEND_WEIGHTS_WITH_BIVARIATE`,
    sonst :data:`DEFAULT_BLEND_WEIGHTS`. Fuer das Standard-Trio ist das Ergebnis
    **identisch** zu den bisherigen Defaults (Default-Stabilitäts-Contract).
    Unbekannte Modellnamen erhalten Gewicht 0; ist die Summe 0, faellt die
    Funktion auf eine Gleichverteilung zurueck (defensiv, nie leere Gewichte).
    """
    names = list(model_names)
    base = BLEND_WEIGHTS_WITH_BIVARIATE if "bivariate" in names else DEFAULT_BLEND_WEIGHTS
    raw = {n: base.get(n, 0.0) for n in names}
    total = sum(raw.values())
    if total <= 0:
        k = len(names) or 1
        return {n: 1.0 / k for n in names}
    return {n: w / total for n, w in raw.items()}


def build_all_goal_models(*, rho: float = 0.1, max_goals: int = 6,
                          negbin_size: float = 8.0,
                          include_bivariate: bool = False,
                          bivariate_lambda3: float = 0.12) -> Dict[str, DixonColesPoisson]:
    """Drei Modelle parallel: poisson, negbin, glm_poisson. Wird vom
    MatchPredictor genutzt, um pro Match alle drei Vorhersagen zu liefern.
    ``include_bivariate=True`` haengt das Karlis-Ntzoufras-Modell als 4. an
    (Phase 4 / Verbesserungsplan 2.2 — opt-in, Default-Output unveraendert)."""
    models = {
        name: build_goal_model(name, rho=rho, max_goals=max_goals, negbin_size=negbin_size)
        for name in MODEL_NAMES
    }
    if include_bivariate:
        models["bivariate"] = build_goal_model(
            "bivariate", max_goals=max_goals, lambda3=bivariate_lambda3)
    return models


def _scalar(markets: Dict[str, Any], key: str) -> float:
    v = markets.get(key, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def blend_markets(
    per_model: Dict[str, Dict[str, Any]],
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Gewichtetes Mittel der skalaren Markets über alle Modelle.

    top_scores werden nicht gemittelt — wir nehmen die des am hoechsten
    gewichteten Modells als Repraesentation, damit Score-Verteilungen
    konsistent bleiben.
    """
    weights = weights or resolve_blend_weights(per_model.keys())
    total_w = sum(weights.get(m, 0.0) for m in per_model)
    if total_w <= 0:
        total_w = 1.0
    out: Dict[str, Any] = {}
    scalar_keys = ["home_win", "draw", "away_win", "over_05", "over_15",
                   "over_25", "over_35", "btts"]
    for key in scalar_keys:
        out[key] = sum(
            _scalar(per_model[m], key) * weights.get(m, 0.0) / total_w
            for m in per_model
        )
    # top_scores aus dem dominanten Modell
    dominant = max(per_model, key=lambda m: weights.get(m, 0.0))
    out["top_scores"] = per_model[dominant].get("top_scores", [])
    return out


def blend_score_matrix(
    models: Dict[str, DixonColesPoisson],
    home_xg: float,
    away_xg: float,
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted average of the per-model score matrices at the same (λ_home, λ_away).

    Because every market we read off the matrix (1X2, totals, BTTS, Asian
    handicap, …) is a *linear* functional of the cells, deriving them from this
    blended matrix yields **exactly** the weighted blend of the per-model market
    values — i.e. the derived markets stay consistent with ``blend_markets`` and
    the heatmap shows the same distribution the headline numbers come from.
    Each ``predict_matrix`` is already normalised to sum 1, and the weights are
    renormalised here, so the result is itself a proper distribution.
    """
    weights = weights or resolve_blend_weights(models.keys())
    total_w = sum(weights.get(name, 0.0) for name in models) or 1.0
    acc: np.ndarray | None = None
    for name, model in models.items():
        w = weights.get(name, 0.0) / total_w
        m = model.predict_matrix(home_xg, away_xg) * w
        acc = m if acc is None else acc + m
    if acc is None:
        return np.zeros((1, 1))
    total = acc.sum()
    return acc / total if total > 0 else acc


def bootstrap_markets(
    model: DixonColesPoisson,
    home_xg: float,
    away_xg: float,
    *,
    n: int = 500,
    xg_sigma: float = 0.15,
    rng: np.random.Generator | None = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Monte-Carlo-CIs durch Sampling von (home_xg', away_xg') ~ Normal.

    Default n=500, xg_sigma=15% des mean. Returns {market: (p5, p50, p95)}.
    Skalare Markets only; top_scores werden ausgelassen.

    Phase 4 (Verbesserungsplan 4.1): Double-Chance-Quantile (``dc_1x``,
    ``dc_12``, ``dc_x2``) werden **pro Sample** als Summe der 1X2-Werte
    akkumuliert — Quantile von Summen, nicht Summen von Quantilen, d.h. die
    Korrelation zwischen den Outcomes bleibt korrekt erfasst. Damit bekommen
    Double-Chance-Edges denselben konservativen p5-Guard wie 1X2/O-U/BTTS.
    Invariante (als Test verankert): ``dc_12 ≡ 1 − draw`` pro Sample ⇒
    ``p5(dc_12) = 1 − p95(draw)``.
    """
    rng = rng or np.random.default_rng()
    home_xg = max(0.05, float(home_xg))
    away_xg = max(0.05, float(away_xg))
    home_samples = np.clip(rng.normal(home_xg, home_xg * xg_sigma, n), 0.05, None)
    away_samples = np.clip(rng.normal(away_xg, away_xg * xg_sigma, n), 0.05, None)

    keys = ["home_win", "draw", "away_win", "over_15", "over_25", "over_35", "btts"]
    dc_keys = ["dc_1x", "dc_12", "dc_x2"]
    accum: Dict[str, list[float]] = {k: [] for k in keys + dc_keys}
    for h, a in zip(home_samples, away_samples):
        m = model.predict_matrix(float(h), float(a))
        mk = model.markets(m)
        for k in keys:
            accum[k].append(_scalar(mk, k))
        hw, dr, aw = (_scalar(mk, "home_win"), _scalar(mk, "draw"),
                      _scalar(mk, "away_win"))
        accum["dc_1x"].append(hw + dr)
        accum["dc_12"].append(hw + aw)
        accum["dc_x2"].append(dr + aw)

    out: Dict[str, Tuple[float, float, float]] = {}
    for k, vals in accum.items():
        arr = np.asarray(vals)
        out[k] = (
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
        )
    return out


def bootstrap_blend_metrics(
    models: Dict[str, DixonColesPoisson],
    home_xg: float,
    away_xg: float,
    fns: Dict[str, Any],
    *,
    weights: Dict[str, float] | None = None,
    n: int = 300,
    xg_sigma: float = 0.15,
    rng: np.random.Generator | None = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap-Quantile beliebiger Matrix-Metriken auf der **geblendeten**
    Score-Matrix (Phase 4 / Verbesserungsplan 4.2).

    ``fns`` ist ``{name: callable(matrix) -> float}``. Pro λ-Sample wird die
    Blend-Matrix **einmal** gebaut und jede Metrik darauf ausgewertet (ein
    Sampling-Durchlauf für alle Metriken). Im Gegensatz zur gewichteten
    Quantil-Mittelung der per-Modell-CIs bootstrappt das den Blend direkt —
    für nichtlineare Funktionale (z.B. Asian-Handicap-No-Push-Anteile) die
    sauberere Verteilungsaussage. Returns ``{name: (p5, p50, p95)}``.
    """
    rng = rng or np.random.default_rng()
    weights = weights or resolve_blend_weights(models.keys())
    home_xg = max(0.05, float(home_xg))
    away_xg = max(0.05, float(away_xg))
    n = max(1, int(n))
    home_samples = np.clip(rng.normal(home_xg, home_xg * xg_sigma, n), 0.05, None)
    away_samples = np.clip(rng.normal(away_xg, away_xg * xg_sigma, n), 0.05, None)

    accum: Dict[str, list[float]] = {name: [] for name in fns}
    for h, a in zip(home_samples, away_samples):
        matrix = blend_score_matrix(models, float(h), float(a), weights)
        for name, fn in fns.items():
            try:
                accum[name].append(float(fn(matrix)))
            except Exception:
                # Eine kaputte Metrik darf die übrigen nicht mitreißen.
                continue

    out: Dict[str, Tuple[float, float, float]] = {}
    for name, vals in accum.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        out[name] = (
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
        )
    return out


__all__ = [
    "DixonColesPoisson",
    "NegativeBinomialDixonColes",
    "BivariatePoisson",
    "build_goal_model",
    "build_all_goal_models",
    "blend_markets",
    "blend_score_matrix",
    "bootstrap_markets",
    "bootstrap_blend_metrics",
    "resolve_blend_weights",
    "MODEL_NAMES",
    "DEFAULT_BLEND_WEIGHTS",
    "BLEND_WEIGHTS_WITH_BIVARIATE",
]
