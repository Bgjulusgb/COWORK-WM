"""Weighted combination of all FactorSignals into λ-multipliers for Dixon-Coles.

Re-normalisation rule:
    If a factor reports available=False, its weight is dropped and the
    remaining weights are scaled so they still sum to 1.0. That way the
    user-configured ratios stay meaningful even when an external source is
    down.

Confidence model (vereinfacht, gem. Spec 3.4):
    0.6 * average(factor_confidence) + 0.4 * agreement
where agreement = 1 - stdev(home_strength_i / away_strength_i).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import structlog

from factors.base import FactorSignal

log = structlog.get_logger("analysis.factor_ensemble")

# Combined global goal-modifier (weather × altitude) can't damp goals by more
# than ~18% — a guardrail against stacked environmental penalties (C2).
_GLOBAL_FLOOR = 0.82


@dataclass
class EnsembleResult:
    lambda_home_multiplier: float          # multiplies base_home_xg
    lambda_away_multiplier: float          # multiplies base_away_xg
    confidence: float                      # 0..1
    used_signals: list[FactorSignal] = field(default_factory=list)
    breakdown_payload: dict[str, Any] = field(default_factory=dict)


class FactorEnsemble:
    """Stateless combiner. Instantiate per-call; no internal mutable state.

    Phase 4 (Verbesserungsplan 2.5): ``aggregation`` steuert, wie die
    Tilt-Faktoren gemittelt werden —
    * ``"arith"`` (Default): gewichtetes arithmetisches Mittel (Bestand),
    * ``"geom"``: log-linear ``exp(Σ wᵢ·ln sᵢ)`` — konsistent fuer
      multiplikative λ-Tilts und symmetrisch in home/away (Kehrwert-Invarianz).
    ``None`` liest ``settings.lambda_aggregation`` (lazy, damit das Modul ohne
    config importierbar bleibt).
    """

    def __init__(self, aggregation: str | None = None) -> None:
        if aggregation is None:
            try:
                from config.settings import settings as _settings
                aggregation = getattr(_settings, "lambda_aggregation", "arith")
            except Exception:  # pragma: no cover - config optional in unit tests
                aggregation = "arith"
        self.aggregation = (aggregation or "arith").lower()

    def combine(self, signals: list[FactorSignal]) -> EnsembleResult:
        # M1: ein post-Validator NaN/inf in home/away_strength wuerde die
        # Lambdas-Summen vergiften. Behandle solche Signale wie not-available.
        def _strengths_ok(s: FactorSignal) -> bool:
            return math.isfinite(s.home_strength) and math.isfinite(s.away_strength)

        available = [s for s in signals if s.available and s.weight > 0 and _strengths_ok(s)]
        total_weight = sum(s.weight for s in available)

        if not available or total_weight <= 0:
            # All factors fell over (or none configured). Leave xG untouched
            # and broadcast zero confidence so the API can flag it.
            for s in signals:
                if not s.available:
                    log.warning("factor_unavailable", name=s.name, source=s.source)
            return EnsembleResult(
                lambda_home_multiplier=1.0,
                lambda_away_multiplier=1.0,
                confidence=0.0,
                used_signals=signals,
                breakdown_payload={
                    "ensemble_confidence": 0.0,
                    "lambda_home_multiplier": 1.0,
                    "lambda_away_multiplier": 1.0,
                    "signals": [_signal_payload(s, effective_weight=0.0) for s in signals],
                    "notes": ["No available factors — predictor falls back to base xG."],
                },
            )

        # Split tilt factors (home/away lean) from global goal modifiers
        # (weather, altitude — symmetric). Averaging a symmetric 0.9 damp in with
        # a dozen ~1.0 factors would dilute it to ~0.99 (inert); instead we
        # average only the tilt factors and multiply the global damp on after.
        tilt = [s for s in available if s.kind != "global"]
        glob = [s for s in available if s.kind == "global"]
        tilt_weight = sum(s.weight for s in tilt)

        if tilt and tilt_weight > 0:
            eff_weights = {s.name: s.weight / tilt_weight for s in tilt}
            if self.aggregation == "geom":
                # Log-lineares Pooling: exp(Σ wᵢ·ln sᵢ). Strengths sind
                # multiplikative Tilts um 1.0; Clamp auf ≥0.05 haelt den Log
                # endlich, falls ein Faktor pathologisch nahe 0 meldet.
                lambda_home = math.exp(sum(
                    eff_weights[s.name] * math.log(max(0.05, s.home_strength))
                    for s in tilt))
                lambda_away = math.exp(sum(
                    eff_weights[s.name] * math.log(max(0.05, s.away_strength))
                    for s in tilt))
            else:
                lambda_home = sum(eff_weights[s.name] * s.home_strength for s in tilt)
                lambda_away = sum(eff_weights[s.name] * s.away_strength for s in tilt)
        else:
            # Only global modifiers present — start neutral, let the damp apply.
            eff_weights = {}
            lambda_home = lambda_away = 1.0

        # Global multiplier = product of the symmetric damps, floored so stacked
        # environmental penalties (hot AND high) can't erase more than ~18% of
        # the goal expectation (C2).
        global_mult = 1.0
        for g in glob:
            global_mult *= g.home_strength
        global_mult = max(_GLOBAL_FLOOR, min(1.05, global_mult))
        lambda_home *= global_mult
        lambda_away *= global_mult

        avg_conf = statistics.fmean(s.confidence for s in available)

        # Agreement: low stdev across the *tilt* factors' home/away ratios →
        # they tell the same story. Global factors carry no tilt, so they're out.
        # M1: epsilon-Guard + NaN/inf-Filter, damit ein post-Validator-mutiertes
        # Signal (Cache-Bug, Mock-Test) die Ensemble nicht zum Crashen bringt.
        ratios: list[float] = []
        for s in tilt:
            num = s.home_strength
            den = s.away_strength
            if not math.isfinite(num) or not math.isfinite(den) or den <= 1e-6:
                ratios.append(1.0)
            else:
                r = num / den
                if math.isfinite(r):
                    ratios.append(r)
                else:
                    ratios.append(1.0)
        if len(ratios) >= 2:
            try:
                spread = statistics.stdev(ratios)
            except statistics.StatisticsError:
                spread = 0.0
        else:
            spread = 0.0
        if not math.isfinite(spread):
            spread = 0.0
        agreement = max(0.0, 1.0 - min(1.0, spread))

        confidence = 0.6 * avg_conf + 0.4 * agreement
        confidence = max(0.0, min(1.0, confidence))

        notes: list[str] = []
        for s in signals:
            if not s.available:
                notes.append(f"{s.name} skipped: {(s.raw_data or {}).get('reason', 'unavailable')}")
        if glob:
            notes.append(f"global goal modifier ×{global_mult:.3f} ({', '.join(g.name for g in glob)})")

        return EnsembleResult(
            lambda_home_multiplier=lambda_home,
            lambda_away_multiplier=lambda_away,
            confidence=confidence,
            used_signals=signals,
            breakdown_payload={
                "ensemble_confidence": confidence,
                "lambda_home_multiplier": lambda_home,
                "lambda_away_multiplier": lambda_away,
                "lambda_aggregation": self.aggregation,
                "agreement": agreement,
                "avg_factor_confidence": avg_conf,
                "global_multiplier": global_mult,
                "signals": [
                    _signal_payload(s, effective_weight=eff_weights.get(s.name, 0.0))
                    for s in signals
                ],
                "notes": notes,
            },
        )


def _signal_payload(s: FactorSignal, effective_weight: float) -> dict[str, Any]:
    """Flatten a FactorSignal to the JSON shape we persist + serve via the API."""
    return {
        "name": s.name,
        "home_strength": s.home_strength,
        "away_strength": s.away_strength,
        "weight": s.weight,
        "effective_weight": effective_weight,
        "confidence": s.confidence,
        "available": s.available,
        "source": s.source,
        "kind": s.kind,
        # Global factors (weather/altitude) damp goals symmetrically rather than
        # tilting a side — the UI renders them as a "goal modifier", not a bar.
        "is_global": s.kind == "global",
        "raw_data": s.raw_data,
        "cached_at": s.cached_at.isoformat() if s.cached_at else None,
        # GoalEfficiencyFactor sets this so the UI can show a "proxy, not real xG" tooltip.
        "is_xg_proxy": bool((s.raw_data or {}).get("is_xg_proxy", False)),
    }


__all__ = ["EnsembleResult", "FactorEnsemble"]
