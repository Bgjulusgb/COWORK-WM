#!/usr/bin/env python3
"""Offline calibration fit — close the data/training gap without the DB or sklearn.

``analysis.calibration.fit_calibrators`` normally reads prediction history from
the database (the admin path). This script lets a fresh clone fit the isotonic +
Platt calibration curves from a flat **CSV of past predictions vs. results**, so
Phase 5 (``--calibrate auto``) activates with core dependencies only.

Build the reference set from a *famous, well-documented* prior — e.g. your
model's predictions for **WC 2022 + EURO 2024 + Copa América 2024** paired with
the actual scorelines — and note the transfer (those tournaments are a prior for
WC 2026, not the same distribution).

CSV columns (header required)::

    home_win_prob,draw_prob,away_win_prob,home_score,away_score
    0.62,0.24,0.14,2,1
    0.31,0.30,0.39,0,0
    ...

Usage::

    python scripts/fit_calibration_offline.py path/to/history.csv
    # → writes models_ml/artifacts/calibration_{isotonic,platt}.json
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

# Allow running as a bare script (python scripts/fit_calibration_offline.py …).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = ("home_win_prob", "draw_prob", "away_win_prob", "home_score", "away_score")


def rows_from_csv(path: str | Path) -> list[SimpleNamespace]:
    """Parse the prediction-history CSV into MatchPrediction-shaped rows.

    Pure + side-effect-free so it is unit-testable. Rows with missing scores or
    unparesable probabilities are skipped (a calibration set is allowed to carry
    not-yet-played fixtures).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calibration CSV not found: {p}")
    rows: list[SimpleNamespace] = []
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for raw in reader:
            try:
                hs = raw.get("home_score")
                as_ = raw.get("away_score")
                if hs in (None, "") or as_ in (None, ""):
                    continue
                rows.append(SimpleNamespace(
                    home_win_prob=float(raw["home_win_prob"]),
                    draw_prob=float(raw["draw_prob"]),
                    away_win_prob=float(raw["away_win_prob"]),
                    actual_home_score=int(float(hs)),
                    actual_away_score=int(float(as_)),
                ))
            except (TypeError, ValueError):
                continue
    return rows


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2

    rows = rows_from_csv(args[0])
    if len(rows) < 5:
        print(f"need ≥5 completed matches to fit, got {len(rows)}", file=sys.stderr)
        return 1

    from analysis.backtesting import compute
    from analysis.calibration import apply, fit_calibrators

    pre = compute(rows)
    iso, platt = fit_calibrators(rows)          # writes the two artifacts

    # Quick before/after sanity: Brier of the raw vs. isotonic-calibrated probs.
    post_sum, n = 0.0, 0
    for r in rows:
        cal = apply(iso, r.home_win_prob, r.draw_prob, r.away_win_prob)
        if cal is None:
            continue
        y = (
            1.0 if r.actual_home_score > r.actual_away_score else 0.0,
            1.0 if r.actual_home_score == r.actual_away_score else 0.0,
            1.0 if r.actual_home_score < r.actual_away_score else 0.0,
        )
        p = (cal["home"], cal["draw"], cal["away"])
        post_sum += sum((pi - yi) ** 2 for pi, yi in zip(p, y))
        n += 1
    post_brier = post_sum / n if n else float("nan")

    print(f"fitted on {len(rows)} matches "
          f"(isotonic n={iso.n_trained_on}, platt n={platt.n_trained_on})")
    print(f"Brier  raw={pre.brier:.4f}  →  isotonic={post_brier:.4f}   "
          f"(RPS raw={pre.rps:.4f})")
    print("→ wrote models_ml/artifacts/calibration_{isotonic,platt}.json — "
          "Phase 5 (--calibrate auto) will now apply them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
