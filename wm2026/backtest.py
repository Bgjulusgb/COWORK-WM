"""Score saved JSON reports against actual match outcomes.

Closes the loop the p5-conservative rule promises: did the conservative pick
hit more often than the raw pick? Reads ``<reports_dir>/*.json`` (the on-disk
artefacts ``wm2026 predict --out`` already writes) and joins them with a CSV
of actual scores keyed by ``match_id``.

Metrics (all on the blended 1X2 line):

* **Brier** — sum-of-squares between predicted triple and outcome one-hot.
* **RPS** — Ranked Probability Score, the order-aware 1X2 metric.
* **LogLoss** — strictly-proper, hard on overconfident wrong predictions.
* hit rate of ``best_value`` (raw max edge) vs ``best_value_cons`` (p5 survivor).
* naïve half-Kelly ROI at the supplied ``bankroll`` (no compounding).

Reuses the metric implementations in :mod:`analysis.backtesting` so any future
change to the math layer propagates automatically.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

from analysis.backtesting import _brier, _log_loss, _outcome_vec, _rps


def read_truth_csv(path: Path) -> dict[str, tuple[int, int]]:
    """Parse a ground-truth CSV → ``{match_id: (home_score, away_score)}``.

    Required columns: ``match_id``, ``home_score``, ``away_score``. Rows with
    missing or unparseable scores are silently dropped so a partial-tournament
    CSV (some fixtures not yet played) is allowed.
    """
    if not path.exists():
        raise FileNotFoundError(f"truth CSV not found: {path}")
    out: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"match_id", "home_score", "away_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"truth CSV missing columns: {sorted(missing)}")
        for raw in reader:
            mid = (raw.get("match_id") or "").strip()
            hs = raw.get("home_score")
            as_ = raw.get("away_score")
            if not mid or hs in (None, "") or as_ in (None, ""):
                continue
            try:
                out[mid] = (int(float(hs)), int(float(as_)))
            except (TypeError, ValueError):
                continue
    return out


def _load_report(path: Path) -> dict[str, Any]:
    """Load a JSON report (transparently handles ``.json`` and ``.json.gz``)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_hit(market: str, selection: str, hs: int, as_: int) -> bool:
    """Did this edge-row selection win given the actual scoreline?

    Covers 1X2, O/U 2.5, BTTS and Double Chance — the markets ``compute_edges``
    emits as universal lines. Unknown markets (e.g. AH or exact scores) return
    False, so they don't pollute hit-rate stats with phantom credit.
    """
    if market == "1X2":
        if selection == "Home":
            return hs > as_
        if selection == "Draw":
            return hs == as_
        if selection == "Away":
            return hs < as_
    if market == "O/U 2.5":
        total = hs + as_
        if selection == "Over 2.5":
            return total > 2
        if selection == "Under 2.5":
            return total < 3
    if market == "BTTS":
        both = hs > 0 and as_ > 0
        if selection == "Yes":
            return both
        if selection == "No":
            return not both
    if market == "Double Chance":
        if selection == "1X":
            return hs >= as_
        if selection == "12":
            return hs != as_
        if selection == "X2":
            return hs <= as_
    return False


def _pnl(row: dict[str, Any] | None, hit: bool, stake_key: str,
         bankroll: float) -> float:
    """Half-Kelly profit/loss for one bet at the stake percentage in ``stake_key``."""
    if not row:
        return 0.0
    stake_pct = row.get(stake_key) or 0.0
    odd = row.get("decimal_odd")
    if stake_pct <= 0 or not odd or float(odd) <= 1.0:
        return 0.0
    stake = bankroll * (float(stake_pct) / 100.0)
    return stake * (float(odd) - 1.0) if hit else -stake


def run_backtest(
    *,
    reports_dir: Path,
    truth_csv: Path,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Iterate every report under ``reports_dir`` and score it against ``truth_csv``.

    Returns a structured dict — see :func:`format_briefing` for the markdown
    rendering. Missing match_ids in the truth CSV are counted but do not
    contribute to any aggregate metric.
    """
    truth = read_truth_csv(truth_csv)
    if not truth:
        raise ValueError(f"no usable rows in truth CSV: {truth_csv}")
    if not reports_dir.exists():
        raise FileNotFoundError(f"reports dir not found: {reports_dir}")

    # Prefer the uncompressed .json; fall back to .json.gz only when its sibling
    # is absent. Without this, a directory that carries both writes would count
    # every match twice (since ``wm2026 predict --gzip`` emits both files).
    json_stems = {p.stem for p in reports_dir.glob("*.json")}
    paths = sorted(reports_dir.glob("*.json"))
    paths += [p for p in sorted(reports_dir.glob("*.json.gz"))
              if p.name.removesuffix(".json.gz") not in json_stems]
    n = 0
    n_missing = 0
    brier_sum = log_sum = rps_sum = 0.0
    correct_1x2 = 0
    bv_hits = bv_n = bvc_hits = bvc_n = 0
    raw_pnl = cons_pnl = 0.0
    per_match: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for p in paths:
        try:
            rep = _load_report(p)
        except Exception:
            continue
        mid = rep.get("match_id")
        if not mid or mid in seen_ids:
            continue
        if mid not in truth:
            n_missing += 1
            continue
        seen_ids.add(mid)
        hs, as_ = truth[mid]
        m1x2 = (rep.get("markets") or {}).get("1x2") or {}
        ph = float(m1x2.get("home", 0.0))
        pd_ = float(m1x2.get("draw", 0.0))
        pa = float(m1x2.get("away", 0.0))
        if (ph + pd_ + pa) <= 0:
            n_missing += 1
            continue

        p_vec = (ph, pd_, pa)
        y_vec = _outcome_vec(hs, as_)
        b = _brier(p_vec, y_vec)
        ll = _log_loss(p_vec, y_vec)
        r = _rps(p_vec, y_vec)
        brier_sum += b
        log_sum += ll
        rps_sum += r

        predicted_idx = max(range(3), key=lambda i: p_vec[i])
        actual_idx = max(range(3), key=lambda i: y_vec[i])
        is_correct = predicted_idx == actual_idx
        if is_correct:
            correct_1x2 += 1

        bv = rep.get("best_value")
        bvc = rep.get("best_value_cons")
        bv_hit = False
        bvc_hit = False
        if bv:
            bv_n += 1
            bv_hit = _selection_hit(bv.get("market", ""), bv.get("selection", ""), hs, as_)
            if bv_hit:
                bv_hits += 1
            raw_pnl += _pnl(bv, bv_hit, "half_kelly_pct", bankroll)
        if bvc:
            bvc_n += 1
            bvc_hit = _selection_hit(bvc.get("market", ""), bvc.get("selection", ""), hs, as_)
            if bvc_hit:
                bvc_hits += 1
            cons_pnl += _pnl(bvc, bvc_hit, "half_kelly_cons", bankroll)

        n += 1
        per_match.append({
            "match_id": mid,
            "actual": f"{hs}-{as_}",
            "predicted_1x2": ["home", "draw", "away"][predicted_idx],
            "correct_1x2": is_correct,
            "brier": round(b, 4),
            "rps": round(r, 4),
            "log_loss": round(ll, 4),
            "best_value": (bv or {}).get("selection"),
            "best_value_hit": bv_hit if bv else None,
            "best_value_cons": (bvc or {}).get("selection"),
            "best_value_cons_hit": bvc_hit if bvc else None,
        })

    def _safe_div(num: float, den: float) -> float | None:
        return round(num / den, 4) if den else None

    def _safe_roi(pnl: float) -> float | None:
        return round(100.0 * pnl / bankroll, 2) if bankroll > 0 else None

    return {
        "n_evaluated": n,
        "n_missing_truth": n_missing,
        "bankroll": bankroll,
        "metrics": {
            "brier":    _safe_div(brier_sum, n),
            "rps":      _safe_div(rps_sum, n),
            "log_loss": _safe_div(log_sum, n),
            "accuracy": _safe_div(correct_1x2, n),
        },
        "best_value": {
            "attempts": bv_n,
            "hits":     bv_hits,
            "hit_rate": _safe_div(bv_hits, bv_n),
            "pnl":      round(raw_pnl, 2),
            "roi_pct":  _safe_roi(raw_pnl),
        },
        "best_value_cons": {
            "attempts": bvc_n,
            "hits":     bvc_hits,
            "hit_rate": _safe_div(bvc_hits, bvc_n),
            "pnl":      round(cons_pnl, 2),
            "roi_pct":  _safe_roi(cons_pnl),
        },
        "per_match": per_match,
    }


def format_briefing(report: dict[str, Any]) -> str:
    """Render a structured ``run_backtest`` result as a terminal-friendly markdown."""
    n = report["n_evaluated"]
    nm = report["n_missing_truth"]
    m = report["metrics"]
    bv = report["best_value"]
    bvc = report["best_value_cons"]

    def _fmt(v: Any) -> str:
        return "—" if v is None else str(v)

    L = [
        "# 📊 WM-2026 Backtest — Reports vs. Reality",
        "",
        f"**{n}** matches evaluated · {nm} reports without matching truth row · "
        f"bankroll {report['bankroll']:.0f}",
        "",
        "## Aggregate accuracy",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Brier (lower=better) | **{_fmt(m['brier'])}** |",
        f"| RPS (lower=better) | **{_fmt(m['rps'])}** |",
        f"| LogLoss (lower=better) | **{_fmt(m['log_loss'])}** |",
        f"| 1X2 best-class hit-rate | **{_fmt(m['accuracy'])}** |",
        "",
        "## Pick hit-rate (raw vs. conservative)",
        "",
        "| pick | attempts | hits | hit-rate | PnL | ROI |",
        "|---|---|---|---|---|---|",
        f"| best_value (raw max edge) | {bv['attempts']} | {bv['hits']} | "
        f"{_fmt(bv['hit_rate'])} | {bv['pnl']:+.2f} | {_fmt(bv['roi_pct'])}% |",
        f"| best_value_cons (p5 survivor) | {bvc['attempts']} | {bvc['hits']} | "
        f"{_fmt(bvc['hit_rate'])} | {bvc['pnl']:+.2f} | {_fmt(bvc['roi_pct'])}% |",
        "",
    ]
    return "\n".join(L)


__all__ = ["read_truth_csv", "run_backtest", "format_briefing"]
