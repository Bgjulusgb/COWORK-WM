"""Phase 6 — Markt-Edge & Value-Detection (de-vigging, edge, Kelly).

Pure functions, zero heavy dependencies. Given a model probability and the
bookmaker's decimal odds, we compute:

* the bookmaker's *fair* (vig-free) probability,
* the model's value **edge** = ``p_model * odd - 1``,
* the **Kelly** stake fraction and the recommended ``half-Kelly`` stake band.

Nothing here touches the network — bookmaker odds come from the match config or
the ``--odds`` CLI flag, exactly the inputs the master prompt's ``match:`` block
provides.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


# ── value bands (master prompt, Phase 6) ──────────────────────────────────────
#   edge < 2%      → no-bet
#   2% .. 5%       → small     (0.25 .. 0.5 % bankroll)
#   5% .. 10%      → standard  (0.5 .. 1.5 % bankroll)
#   > 10%          → flag for a sanity check first, then max 2 % bankroll
def stake_band(edge: float) -> str:
    if edge < 0.02:
        return "no-bet"
    if edge < 0.05:
        return "small"
    if edge < 0.10:
        return "standard"
    return "sanity-check"


@dataclass
class EdgeRow:
    """One market line evaluated against the book."""
    market: str
    selection: str
    model_p: float
    fair_p: float | None          # vig-free book probability (None if no odds)
    decimal_odd: float | None
    edge_pct: float | None        # (model_p * odd - 1) * 100
    kelly_pct: float | None       # full-Kelly stake as % of bankroll
    half_kelly_pct: float | None  # recommended (half-Kelly)
    action: str
    # Conservative (risk-managed) staking: edge & half-Kelly recomputed on the
    # bootstrap *lower bound* (p5) of the model probability instead of the point
    # estimate. None when no confidence interval was supplied. A positive
    # conservative edge means the bet survives the 5th-percentile of the model's
    # own uncertainty — a much stronger value signal than the p50 edge alone.
    model_p_lower: float | None = None
    edge_pct_cons: float | None = None
    half_kelly_cons: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_odds(spec: str | None) -> list[float] | None:
    """Parse a ``"2.10 / 3.40 / 3.20"`` style string into ``[2.10, 3.40, 3.20]``.

    Accepts ``/``, ``,`` or whitespace as separators. Returns ``None`` for an
    empty/placeholder spec so callers can treat "no odds given" uniformly.
    """
    if not spec:
        return None
    cleaned = spec.replace("/", " ").replace(",", " ")
    out: list[float] = []
    for tok in cleaned.split():
        try:
            val = float(tok)
        except ValueError:
            continue
        if val > 1.0:                       # decimal odds are always > 1.0
            out.append(val)
    return out or None


def devig(decimal_odds: list[float]) -> tuple[list[float], float]:
    """Remove the bookmaker's overround → (fair_probabilities, overround).

    ``overround`` (a.k.a. vig/juice) is ``sum(1/odd)`` and is > 1.0 for any
    book that makes money. Dividing the raw implied probabilities by it yields
    the vig-free estimate of the book's true probabilities.

    Malformed-input contract: only entries with ``o > 1.0`` are valid decimal
    odds and survive the filter. If every input is invalid (e.g. ``[0, 0, 0]``
    from a typo or ``[-1, 0, 1]`` from a copy-paste error) the function returns
    ``([], 0.0)`` so callers can detect "no usable odds" without an IndexError.
    Partial input (one valid odd out of three for 1X2) returns fewer entries
    than were given — callers that need positional alignment MUST check
    ``len(fair)`` before indexing (every site in :func:`compute_edges` already
    does so via ``len(fair_*) > i`` guards).
    """
    implied = [1.0 / o for o in decimal_odds if o and o > 1.0]
    overround = sum(implied)
    if overround <= 0:
        return [], 0.0
    fair = [p / overround for p in implied]
    return fair, overround


def kelly_fraction(model_p: float, decimal_odd: float) -> float:
    """Full-Kelly stake fraction for a single bet. Clamped at 0 (no negative
    stakes — a negative Kelly just means "don't bet")."""
    b = decimal_odd - 1.0
    if b <= 0:
        return 0.0
    f = (model_p * decimal_odd - 1.0) / b
    return max(0.0, f)


def evaluate_line(
    market: str,
    selection: str,
    model_p: float,
    decimal_odd: float | None,
    fair_p: float | None = None,
    *,
    model_p_lower: float | None = None,
) -> EdgeRow:
    """Build one :class:`EdgeRow`. ``decimal_odd=None`` ⇒ model-only row.

    ``model_p_lower`` (the bootstrap p5 of this probability) enables the
    conservative edge / half-Kelly columns.
    """
    if decimal_odd is None or decimal_odd <= 1.0:
        return EdgeRow(
            market=market, selection=selection, model_p=round(model_p, 4),
            fair_p=round(fair_p, 4) if fair_p is not None else None,
            decimal_odd=None, edge_pct=None, kelly_pct=None,
            half_kelly_pct=None, action="—",
            model_p_lower=round(model_p_lower, 4) if model_p_lower is not None else None,
        )
    edge = model_p * decimal_odd - 1.0
    full_kelly = kelly_fraction(model_p, decimal_odd)
    band = stake_band(edge)
    edge_cons = half_kelly_cons = None
    if model_p_lower is not None:
        edge_cons = round((model_p_lower * decimal_odd - 1.0) * 100.0, 2)
        half_kelly_cons = round(0.5 * kelly_fraction(model_p_lower, decimal_odd) * 100.0, 2)
    return EdgeRow(
        market=market,
        selection=selection,
        model_p=round(model_p, 4),
        fair_p=round(fair_p, 4) if fair_p is not None else None,
        decimal_odd=round(decimal_odd, 3),
        edge_pct=round(edge * 100.0, 2),
        kelly_pct=round(full_kelly * 100.0, 2),
        half_kelly_pct=round(0.5 * full_kelly * 100.0, 2),
        action=band,
        model_p_lower=round(model_p_lower, 4) if model_p_lower is not None else None,
        edge_pct_cons=edge_cons,
        half_kelly_cons=half_kelly_cons,
    )


def _lower(
    ci: dict[str, Any] | None, key: str, *, complement: bool = False
) -> float | None:
    """5th-percentile of a market probability from the bootstrap CI triple.

    For a *complement* selection (Under = 1−Over, No = 1−Yes) the conservative
    lower bound is ``1 − p95`` of the positive side, not its own p5.
    """
    if not ci:
        return None
    triple = ci.get(key)
    if not triple or len(triple) < 3:
        return None
    return float(1.0 - triple[2]) if complement else float(triple[0])


def compute_edges(
    markets: dict[str, float],
    *,
    odds_1x2: list[float] | None = None,
    odds_ou25: list[float] | None = None,   # [over, under]
    odds_btts: list[float] | None = None,   # [yes, no]
    odds_dc: list[float] | None = None,     # [1X, 12, X2] double chance
    ci: dict[str, Any] | None = None,       # {market: [p5, p50, p95]} for conservative staking
) -> list[dict[str, Any]]:
    """Phase 6 — evaluate every market for which odds were supplied.

    ``markets`` is the blended model output (``home_win``/``draw``/``away_win``/
    ``over_25``/``btts`` …). Only lines with supplied odds yield an edge; the
    rest are still listed (model-only) so the report shows the full picture.
    ``ci`` (the blended bootstrap intervals) unlocks the conservative edge /
    half-Kelly columns — staking that survives the model's own p5 uncertainty.
    """
    rows: list[EdgeRow] = []

    # 1X2 ---------------------------------------------------------------------
    fair_1x2: list[float] | None = None
    if odds_1x2 and len(odds_1x2) >= 3:
        fair_1x2, _ = devig(odds_1x2[:3])
    for i, (sel, key) in enumerate(
        [("Home", "home_win"), ("Draw", "draw"), ("Away", "away_win")]
    ):
        rows.append(evaluate_line(
            "1X2", sel, markets.get(key, 0.0),
            odds_1x2[i] if odds_1x2 and len(odds_1x2) > i else None,
            fair_1x2[i] if fair_1x2 and len(fair_1x2) > i else None,
            model_p_lower=_lower(ci, key),
        ))

    # Over/Under 2.5 ----------------------------------------------------------
    over = markets.get("over_25", 0.0)
    fair_ou: list[float] | None = None
    if odds_ou25 and len(odds_ou25) >= 2:
        fair_ou, _ = devig(odds_ou25[:2])
    rows.append(evaluate_line(
        "O/U 2.5", "Over 2.5", over,
        odds_ou25[0] if odds_ou25 else None,
        fair_ou[0] if fair_ou else None,
        model_p_lower=_lower(ci, "over_25"),
    ))
    rows.append(evaluate_line(
        "O/U 2.5", "Under 2.5", 1.0 - over,
        odds_ou25[1] if odds_ou25 and len(odds_ou25) > 1 else None,
        fair_ou[1] if fair_ou and len(fair_ou) > 1 else None,
        model_p_lower=_lower(ci, "over_25", complement=True),
    ))

    # BTTS --------------------------------------------------------------------
    btts = markets.get("btts", 0.0)
    fair_btts: list[float] | None = None
    if odds_btts and len(odds_btts) >= 2:
        fair_btts, _ = devig(odds_btts[:2])
    rows.append(evaluate_line(
        "BTTS", "Yes", btts,
        odds_btts[0] if odds_btts else None,
        fair_btts[0] if fair_btts else None,
        model_p_lower=_lower(ci, "btts"),
    ))
    rows.append(evaluate_line(
        "BTTS", "No", 1.0 - btts,
        odds_btts[1] if odds_btts and len(odds_btts) > 1 else None,
        fair_btts[1] if fair_btts and len(fair_btts) > 1 else None,
        model_p_lower=_lower(ci, "btts", complement=True),
    ))

    # Double Chance (optional) ------------------------------------------------
    # DC outcomes overlap, so there's no clean 3-way de-vig — we report the
    # model probability + raw edge/Kelly. Phase 4 (4.1): the conservative p5
    # bound comes from the bootstrap's per-sample DC sums (dc_1x/dc_12/dc_x2),
    # so a DC "standard" call now has to survive the same p5 discipline as 1X2.
    if odds_dc:
        h = markets.get("home_win", 0.0)
        d = markets.get("draw", 0.0)
        a = markets.get("away_win", 0.0)
        for i, (sel, p, ci_key) in enumerate(
            [("1X", h + d, "dc_1x"), ("12", h + a, "dc_12"), ("X2", d + a, "dc_x2")]
        ):
            rows.append(evaluate_line(
                "Double Chance", sel, p,
                odds_dc[i] if len(odds_dc) > i else None,
                model_p_lower=_lower(ci, ci_key),
            ))

    return [r.as_dict() for r in rows]


def evaluate_asian_handicap(
    ah: dict[str, float],
    *,
    home_odd: float | None = None,
    away_odd: float | None = None,
    home_p_lower: float | None = None,
    away_p_lower: float | None = None,
) -> list[dict[str, Any]]:
    """Two :class:`EdgeRow`s for an Asian-handicap line from its model shares.

    ``ah`` is one entry of ``markets.asian_handicap`` (keys ``line``,
    ``home_win``, ``push``, ``away_win`` as *expected shares*, plus the
    ``*_prob_nopush`` display probabilities). The edge is the true push-aware
    EV — ``EV(back @ odd o) = win·o + push − 1`` — while the displayed model_p
    and the Kelly stake use the no-push-adjusted probability so the bet sits on
    the same axis as a clean two-way market.

    Phase 4 (4.2): ``home_p_lower``/``away_p_lower`` are the bootstrap p5 of the
    *no-push* probabilities (from ``bootstrap_blend_metrics``). The conservative
    edge/half-Kelly use the same desk approximation as the point estimate
    (no-push prob as a two-way market), so the p5 column is on the same axis as
    the displayed ``model_p``.
    """
    line = ah.get("line", 0.0)
    sign = "+" if line >= 0 else ""
    label = f"AH {sign}{line:g}"
    rows: list[EdgeRow] = []
    sides = [
        ("Home", ah.get("home_win", 0.0), ah.get("home_prob_nopush", 0.0),
         home_odd, home_p_lower),
        ("Away", ah.get("away_win", 0.0), ah.get("away_prob_nopush", 0.0),
         away_odd, away_p_lower),
    ]
    push = ah.get("push", 0.0)
    for sel, win_share, prob_np, odd, p_lower in sides:
        if odd is None or odd <= 1.0:
            rows.append(EdgeRow(
                market=label, selection=sel, model_p=round(prob_np, 4),
                fair_p=None, decimal_odd=None, edge_pct=None, kelly_pct=None,
                half_kelly_pct=None, action="—",
                model_p_lower=round(p_lower, 4) if p_lower is not None else None,
            ))
            continue
        ev = win_share * odd + push - 1.0       # push-aware expected value
        full_kelly = kelly_fraction(prob_np, odd)   # desk approximation (push lowers variance)
        edge_cons = half_kelly_cons = None
        if p_lower is not None:
            edge_cons = round((p_lower * odd - 1.0) * 100.0, 2)
            half_kelly_cons = round(0.5 * kelly_fraction(p_lower, odd) * 100.0, 2)
        rows.append(EdgeRow(
            market=label, selection=sel, model_p=round(prob_np, 4),
            fair_p=None, decimal_odd=round(odd, 3),
            edge_pct=round(ev * 100.0, 2),
            kelly_pct=round(full_kelly * 100.0, 2),
            half_kelly_pct=round(0.5 * full_kelly * 100.0, 2),
            action=stake_band(ev),
            model_p_lower=round(p_lower, 4) if p_lower is not None else None,
            edge_pct_cons=edge_cons,
            half_kelly_cons=half_kelly_cons,
        ))
    return [r.as_dict() for r in rows]


def best_value_pick(edge_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-edge actionable row (edge ≥ 2 %), or ``None``."""
    actionable = [
        r for r in edge_rows
        if r.get("edge_pct") is not None and r["edge_pct"] >= 2.0
    ]
    if not actionable:
        return None
    return max(actionable, key=lambda r: r["edge_pct"])


def best_value_cons_pick(edge_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Phase 4 (4.3) — the *honest* pick: highest **conservative** (p5) edge.

    ``best_value_pick`` maximises the raw p50 edge and therefore tends to
    surface exactly the sanity-check candidates (big edge, thin data). This
    companion only considers rows whose edge stays positive on the bootstrap
    lower bound — the value that survives the model's own uncertainty. Returns
    ``None`` when no row passes (the briefing then says "Pass" honestly).
    """
    actionable = [
        r for r in edge_rows
        if r.get("edge_pct_cons") is not None and r["edge_pct_cons"] > 0.0
    ]
    if not actionable:
        return None
    return max(actionable, key=lambda r: r["edge_pct_cons"])


__all__ = [
    "EdgeRow", "parse_odds", "devig", "kelly_fraction",
    "evaluate_line", "compute_edges", "evaluate_asian_handicap",
    "best_value_pick", "best_value_cons_pick", "stake_band",
]
