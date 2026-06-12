"""Dixon-Coles MLE of team attack/defence strengths from weighted history.

A genuinely model-based replacement for the naive ``_base_xg`` average: fit
per-team attack and defence effects plus a home advantage by maximum likelihood
over a team's recent matches, **time-decayed** with the Dixon-Coles weight
``w = exp(-ξ·Δt_days)·tier_weight`` (ξ=0.0065/day ≈ a ~2-year half-life is the
original Dixon & Coles 1997 value; shorter half-lives weight recent form more).

The model (independent Dixon-Coles Poisson with team effects on the log mean)::

    log λ_home = μ + home_adv + attack[home] − defence[away]
    log λ_away = μ            + attack[away] − defence[home]
    ℓ(m) = logPois(h | λ_home) + logPois(a | λ_away) + log τ_ρ(h, a, λ_home, λ_away)

Attack/defence are made identifiable with the sum-to-zero constraints
``Σ attack = Σ defence = 0`` (μ absorbs the global goal level, home_adv the home
tilt). ρ is held fixed (default 0.1, the scoring model's value) for stability.

Everything here is **gated off by default** (``settings.use_mle_xg = False``) and
falls back hard to the YAML path whenever the fit is unidentifiable, sparse, or
produces non-physical λ — so the default pipeline output is unchanged. Pure
numpy/scipy; no new dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from factors._history import _TIER_WEIGHT, _is_finished


@dataclass
class TeamStrengths:
    attack: dict[str, float]
    defence: dict[str, float]
    home_adv: float
    mu: float
    rho: float
    n_matches: int
    converged: bool
    teams: list[str] = field(default_factory=list)


def _dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles low-score correction τ — identical to
    ``models_ml.poisson_goals.DixonColesPoisson._correction`` so the estimator
    and the scoring model agree."""
    if rho == 0.0:
        return 1.0
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _as_utc(dt: Any) -> datetime | None:
    if not isinstance(dt, datetime):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _collect_games(matches: Sequence[Any], ref_time: datetime, xi: float):
    """→ list of (home_code, away_code, gh, ga, weight), deduped by (codes, date)."""
    ref = _as_utc(ref_time) or datetime.now(timezone.utc)
    seen: set[tuple] = set()
    games: list[tuple[str, str, int, int, float]] = []
    for m in matches:
        if not _is_finished(m):
            continue
        h = (getattr(m, "home_code", "") or "").upper()
        a = (getattr(m, "away_code", "") or "").upper()
        if not h or not a or h == a:
            continue
        try:
            gh, ga = int(m.home_score), int(m.away_score)
        except (TypeError, ValueError):
            continue
        ko = _as_utc(getattr(m, "kickoff_utc", None))
        dt_days = max(0.0, (ref - ko).total_seconds() / 86400.0) if ko else 0.0
        key = (h, a, gh, ga, round(dt_days))
        if key in seen:
            continue
        seen.add(key)
        tier = _TIER_WEIGHT.get(getattr(m, "competition_tier", 4), 0.5)
        games.append((h, a, gh, ga, math.exp(-xi * dt_days) * tier))
    return games


def estimate_strengths(
    matches: Sequence[Any],
    *,
    ref_time: datetime,
    xi: float = 0.0065,
    rho0: float = 0.1,
    min_matches: int = 6,
    max_iter: int = 200,
) -> TeamStrengths | None:
    """Fit attack/defence/home-advantage by weighted MLE. Returns ``None`` when
    the data is too sparse or non-identifiable (caller falls back to YAML)."""
    games = _collect_games(matches, ref_time, xi)
    if len(games) < min_matches:
        return None

    teams = sorted({g[0] for g in games} | {g[1] for g in games})
    T = len(teams)
    if T < 2:
        return None
    idx = {t: i for i, t in enumerate(teams)}

    # Identifiability: every team needs ≥ 2 distinct opponents.
    opponents: dict[str, set[str]] = {t: set() for t in teams}
    for h, a, *_ in games:
        opponents[h].add(a)
        opponents[a].add(h)
    if min(len(o) for o in opponents.values()) < 2:
        return None

    gi = np.array([(idx[h], idx[a], gh, ga) for h, a, gh, ga, _ in games], dtype=float)
    w = np.array([g[4] for g in games], dtype=float)
    hi = gi[:, 0].astype(int); ai = gi[:, 1].astype(int)
    gh = gi[:, 2]; ga = gi[:, 3]
    lgam_h = np.array([math.lgamma(x + 1) for x in gh])
    lgam_a = np.array([math.lgamma(x + 1) for x in ga])
    mean_goals = max(0.3, float((gh + ga).mean()) / 2.0)

    def _unpack(x):
        mu, hadv = x[0], x[1]
        a = np.empty(T); a[:T - 1] = x[2:2 + T - 1]; a[-1] = -a[:T - 1].sum()
        d = np.empty(T); d[:T - 1] = x[2 + T - 1:2 + 2 * (T - 1)]; d[-1] = -d[:T - 1].sum()
        return mu, hadv, a, d

    def _nll(x):
        mu, hadv, a, d = _unpack(x)
        log_lh = np.clip(mu + hadv + a[hi] - d[ai], -6.0, 3.0)
        log_la = np.clip(mu + a[ai] - d[hi], -6.0, 3.0)
        lh = np.exp(log_lh); la = np.exp(log_la)
        ll = (gh * log_lh - lh - lgam_h) + (ga * log_la - la - lgam_a)
        if rho0 != 0.0:                     # vectorised τ only touches the 4 low scores
            tau = np.ones_like(lh)
            m00 = (gh == 0) & (ga == 0); tau[m00] = 1.0 - lh[m00] * la[m00] * rho0
            m01 = (gh == 0) & (ga == 1); tau[m01] = 1.0 + lh[m01] * rho0
            m10 = (gh == 1) & (ga == 0); tau[m10] = 1.0 + la[m10] * rho0
            m11 = (gh == 1) & (ga == 1); tau[m11] = 1.0 - rho0
            ll = ll + np.log(np.clip(tau, 1e-6, None))
        return -float((w * ll).sum())

    x0 = np.concatenate([[math.log(mean_goals), 0.15], np.zeros(2 * (T - 1))])
    try:
        res = minimize(_nll, x0, method="L-BFGS-B", options={"maxiter": max_iter})
    except Exception:
        return None
    if not np.all(np.isfinite(res.x)):
        return None

    mu, hadv, a, d = _unpack(res.x)
    return TeamStrengths(
        attack={teams[i]: float(a[i]) for i in range(T)},
        defence={teams[i]: float(d[i]) for i in range(T)},
        home_adv=float(hadv), mu=float(mu), rho=rho0,
        n_matches=len(games), converged=bool(res.success), teams=teams,
    )


def lambdas_for_fixture(s: TeamStrengths, home_code: str, away_code: str) -> tuple[float, float]:
    """(λ_home, λ_away) for a fixture under the fitted strengths."""
    h, a = home_code.upper(), away_code.upper()
    log_lh = s.mu + s.home_adv + s.attack.get(h, 0.0) - s.defence.get(a, 0.0)
    log_la = s.mu + s.attack.get(a, 0.0) - s.defence.get(h, 0.0)
    return math.exp(log_lh), math.exp(log_la)


def estimate_base_xg(ctx: Any, home_code: str, away_code: str, *, settings: Any):
    """Guarded entry point used by the pipeline. Returns ``((λh, λa) | None, diag)``;
    ``None`` ⇒ caller keeps the YAML ``_base_xg``. Default-off via ``use_mle_xg``."""
    if not getattr(settings, "use_mle_xg", False):
        return None, {"source": "yaml", "note": "use_mle_xg disabled"}
    matches = (list(getattr(ctx, "historical_matches_home", []))
               + list(getattr(ctx, "historical_matches_away", []))
               + list(getattr(ctx, "head_to_head", [])))
    s = estimate_strengths(
        matches, ref_time=getattr(ctx, "kickoff_utc", datetime.now(timezone.utc)),
        xi=getattr(settings, "mle_time_decay_xi", 0.0065),
        min_matches=getattr(settings, "mle_min_matches", 6),
        max_iter=getattr(settings, "mle_max_iter", 200),
    )
    if s is None:
        return None, {"source": "yaml", "note": "insufficient / non-identifiable history"}
    h, a = home_code.upper(), away_code.upper()
    if h not in s.attack or a not in s.attack:
        return None, {"source": "yaml", "note": "fixture team absent from fitted history"}
    lh, la = lambdas_for_fixture(s, h, a)
    if not (math.isfinite(lh) and math.isfinite(la)) or not (0.2 <= lh <= 4.5 and 0.2 <= la <= 4.5):
        return None, {"source": "yaml", "note": "MLE λ out of sane range"}
    return (lh, la), {"source": "mle", "n_matches": s.n_matches,
                      "home_adv": round(s.home_adv, 3), "converged": s.converged}


__all__ = ["TeamStrengths", "estimate_strengths", "lambdas_for_fixture", "estimate_base_xg"]
