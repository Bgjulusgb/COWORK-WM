"""Tournament Monte-Carlo — group stage → knockout, advancement & title odds.

Simulates the whole bracket many times by **sampling scorelines from the blended
score matrix**, ranking the round-robin groups, seeding qualifiers into a
single-elimination knockout, and recursing to the final. Aggregates per-team
``P(reach knockout)``, ``P(reach final)`` and ``P(title)`` over ``n_sims`` runs.

Speed (the design that makes 10k sims of the full 48-team field run in seconds):
* the **group stage is vectorised** — every group match is sampled for all sims
  at once via one ``searchsorted`` on the flattened score-matrix CDF;
* the **knockout** needs only *who advances*, so each pairing's
  ``P(a beats b)`` (incl. the drawn-match coin-flip) is precomputed/cached and a
  KO match is a single Bernoulli draw — no per-sim matrix building, and
  ``run_prediction`` is never called in the loop (λ comes from ``lam_provider``).

The knockout uses standard **seeded** single-elimination over the qualifiers
(group winners seeded above runners-up above best-thirds). The exact FIFA slot
map is a deliberate, documented simplification. Pure numpy — no new deps.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

LamProvider = Callable[[str, str], "tuple[float, float]"]


@dataclass
class TournamentResult:
    advance_prob: dict[str, float]     # P(reach the knockout stage)
    final_prob: dict[str, float]       # P(reach the final)
    title_prob: dict[str, float]       # P(win the tournament)
    n_sims: int

    def ranked(self, key: str = "title_prob") -> list[tuple[str, float]]:
        return sorted(getattr(self, key).items(), key=lambda kv: kv[1], reverse=True)


def _bracket_seeds(n: int) -> list[int]:
    """Standard single-elimination seeding order (1-indexed) for a field of n=2^k:
    seed 1 and seed 2 can only meet in the final."""
    seeds = [1]
    while len(seeds) < n:
        m = len(seeds) * 2 + 1
        seeds = [x for s in seeds for x in (s, m - s)]
    return seeds


def _largest_pow2(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _win_prob_from_matrix(m: np.ndarray) -> float:
    """P(home beats away) including the knockout coin-flip on a drawn scoreline
    (home advances with prob P(win)/(P(win)+P(loss)))."""
    ph = float(np.tril(m, -1).sum())
    pa = float(np.triu(m, 1).sum())
    pd = 1.0 - ph - pa
    denom = ph + pa
    return ph + (pd * ph / denom if denom > 1e-12 else 0.5 * pd)


def simulate_tournament(
    groups: dict[str, list[str]],
    *,
    lam_provider: LamProvider,
    models,
    weights=None,
    n_sims: int = 10_000,
    seed: int = 0,
    n_best_thirds: int | None = None,
) -> TournamentResult:
    """Run ``n_sims`` full-tournament simulations.

    ``groups`` maps a group name → its team codes (round-robin within the group).
    Top-2 of each group advance; the ``n_best_thirds`` best third-placed teams
    also qualify (default fills the next power-of-two knockout field — the
    WC-2026 "8 best thirds" for 12 groups). ``lam_provider`` returns the (neutral)
    goal expectations for a pairing.
    """
    from models_ml.poisson_goals import blend_score_matrix

    rng = np.random.default_rng(seed)
    teams = [t for ts in groups.values() for t in ts]
    tidx = {t: i for i, t in enumerate(teams)}
    ntot = len(teams)
    gnames = list(groups.keys())

    matrix_cache: dict[tuple[str, str], np.ndarray] = {}

    def matrix_for(a: str, b: str) -> np.ndarray:
        m = matrix_cache.get((a, b))
        if m is None:
            lh, la = lam_provider(a, b)
            m = np.asarray(blend_score_matrix(models, lh, la, weights))
            matrix_cache[(a, b)] = m
        return m

    wcache: dict[tuple[int, int], float] = {}

    def win_prob(i: int, j: int) -> float:
        w = wcache.get((i, j))
        if w is None:
            w = _win_prob_from_matrix(matrix_for(teams[i], teams[j]))
            wcache[(i, j)] = w
        return w

    if n_best_thirds is None:
        base = 2 * len(gnames)
        n_best_thirds = max(0, min(len(gnames), _largest_pow2(base + len(gnames)) - base))

    # ── vectorised group stage → per-group 1st/2nd/3rd team indices (n_sims,) ──
    firsts_c, seconds_c, thirds_c, third_score_c = [], [], [], []
    for gteams in groups.values():
        m = len(gteams)
        pts = np.zeros((n_sims, m)); gd = np.zeros((n_sims, m)); gf = np.zeros((n_sims, m))
        for ia, ib in itertools.combinations(range(m), 2):
            mat = matrix_for(gteams[ia], gteams[ib])
            cdf = np.cumsum(mat.ravel()); n = mat.shape[0]
            idx = np.searchsorted(cdf, rng.random(n_sims) * cdf[-1])
            np.clip(idx, 0, n * n - 1, out=idx)
            ga, gb = idx // n, idx % n
            gf[:, ia] += ga; gf[:, ib] += gb
            gd[:, ia] += ga - gb; gd[:, ib] += gb - ga
            wa, wb = ga > gb, gb > ga; dr = ~(wa | wb)
            pts[:, ia] += wa * 3 + dr; pts[:, ib] += wb * 3 + dr
        score = pts * 1e6 + (gd + 1000.0) * 100.0 + gf      # composite rank key
        order = np.argsort(-score, axis=1, kind="stable")
        gidx = np.array([tidx[t] for t in gteams])
        firsts_c.append(gidx[order[:, 0]])
        seconds_c.append(gidx[order[:, 1]])
        if m >= 3:
            thirds_c.append(gidx[order[:, 2]])
            third_score_c.append(np.take_along_axis(score, order[:, 2:3], axis=1)[:, 0])

    firsts = np.stack(firsts_c, axis=1)
    seconds = np.stack(seconds_c, axis=1)
    use_thirds = n_best_thirds > 0 and len(thirds_c) == len(gnames)
    if use_thirds:
        thirds = np.stack(thirds_c, axis=1)
        third_scores = np.stack(third_score_c, axis=1)

    # ── per-sim knockout via cached win probabilities ──
    advance = np.zeros(ntot); final = np.zeros(ntot); title = np.zeros(ntot)
    qual_size = 2 * len(gnames) + (n_best_thirds if use_thirds else 0)
    field = _largest_pow2(qual_size)
    seed_order = _bracket_seeds(field)
    rand = rng.random

    for s in range(n_sims):
        qual = list(firsts[s]) + list(seconds[s])
        if use_thirds:
            best = np.argsort(-third_scores[s])[:n_best_thirds]
            qual += list(thirds[s][best])
        for q in qual:
            advance[int(q)] += 1
        bracket = [int(qual[seed_order[k] - 1]) for k in range(field)]
        cur = bracket
        while len(cur) > 1:
            if len(cur) == 2:
                final[cur[0]] += 1; final[cur[1]] += 1
            cur = [(cur[k] if rand() < win_prob(cur[k], cur[k + 1]) else cur[k + 1])
                   for k in range(0, len(cur), 2)]
        title[cur[0]] += 1

    inv = 1.0 / n_sims
    return TournamentResult(
        advance_prob={teams[i]: advance[i] * inv for i in range(ntot)},
        final_prob={teams[i]: final[i] * inv for i in range(ntot)},
        title_prob={teams[i]: title[i] * inv for i in range(ntot)},
        n_sims=n_sims,
    )


__all__ = ["TournamentResult", "simulate_tournament", "_bracket_seeds", "_largest_pow2"]
