"""Phase 4 (Verbesserungsplan 2.5) — geometrische λ-Aggregation im Ensemble.

Bare-pytest by design (kein pytest-asyncio); FactorSignals werden synthetisch
gebaut, keine Netz-/Settings-Abhängigkeit (aggregation explizit übergeben).
"""
from __future__ import annotations

import math

from analysis.factor_ensemble import FactorEnsemble
from factors.base import FactorSignal


def _sig(name: str, home: float, away: float, weight: float = 1.0,
         conf: float = 0.8, kind: str = "tilt") -> FactorSignal:
    return FactorSignal(
        name=name, home_strength=home, away_strength=away,
        weight=weight, confidence=conf, available=True,
        source="test", kind=kind,
    )


def test_geom_equal_weights_is_geometric_mean():
    # exp(0.5·ln 2 + 0.5·ln 0.5) = 1.0 — das arithmetische Mittel wäre 1.25.
    signals = [_sig("a", 2.0, 2.0), _sig("b", 0.5, 0.5)]
    geom = FactorEnsemble(aggregation="geom").combine(signals)
    arith = FactorEnsemble(aggregation="arith").combine(signals)
    assert abs(geom.lambda_home_multiplier - 1.0) < 1e-9
    assert abs(arith.lambda_home_multiplier - 1.25) < 1e-9


def test_geom_is_home_away_symmetric():
    """Kehrwert-Invarianz: vertauschte Strengths ⇒ exakt invertierte Multiplikatoren.

    Das ist die Eigenschaft, die das arithmetische Mittel verletzt (Mittel von
    s und Mittel von 1/s sind keine Kehrwerte voneinander).
    """
    signals = [_sig("elo", 1.30, 0.85), _sig("form", 0.90, 1.10, weight=0.5)]
    mirrored = [_sig("elo", 0.85, 1.30), _sig("form", 1.10, 0.90, weight=0.5)]
    g = FactorEnsemble(aggregation="geom")
    res = g.combine(signals)
    res_m = g.combine(mirrored)
    assert abs(res.lambda_home_multiplier - res_m.lambda_away_multiplier) < 1e-12
    assert abs(res.lambda_away_multiplier - res_m.lambda_home_multiplier) < 1e-12
    # Und multiplikative Konsistenz: home·(gespiegelt home) == home·away-Produkt
    # bleibt unter Spiegelung erhalten.
    assert abs(
        res.lambda_home_multiplier * res.lambda_away_multiplier
        - res_m.lambda_home_multiplier * res_m.lambda_away_multiplier
    ) < 1e-12


def test_default_aggregation_is_arith_and_unchanged():
    # Default-Stabilitäts-Contract: ohne explizites Setting bleibt das
    # arithmetische Mittel aktiv und liefert die historischen Zahlen.
    signals = [_sig("a", 1.2, 0.9, weight=0.25), _sig("b", 1.0, 1.1, weight=0.75)]
    default = FactorEnsemble().combine(signals)
    arith = FactorEnsemble(aggregation="arith").combine(signals)
    assert default.breakdown_payload["lambda_aggregation"] == "arith"
    assert abs(default.lambda_home_multiplier - arith.lambda_home_multiplier) < 1e-12
    # Erwartung von Hand: w_a=0.25, w_b=0.75 → home = .25·1.2 + .75·1.0 = 1.05
    assert abs(default.lambda_home_multiplier - 1.05) < 1e-9


def test_geom_handles_extreme_strengths_finite():
    # FactorSignal clamped Strength 0.0 → 0.3 (Validator); der Log bleibt
    # endlich und das Ergebnis ist die geometrische Erwartung sqrt(0.3).
    # (Der zusätzliche 0.05-Clamp im Ensemble ist Belt-and-Braces für
    # nicht-validierte Signalquellen.)
    signals = [_sig("broken", 0.0, 1.0), _sig("ok", 1.0, 1.0)]
    res = FactorEnsemble(aggregation="geom").combine(signals)
    assert math.isfinite(res.lambda_home_multiplier)
    assert res.lambda_home_multiplier > 0.0
    assert abs(res.lambda_home_multiplier - math.sqrt(0.3)) < 1e-9


def test_geom_respects_global_modifier_and_payload():
    # Globale Dämpfer (Wetter/Höhe) multiplizieren wie gehabt NACH der
    # Tilt-Aggregation — unabhängig vom Aggregationsmodus.
    signals = [
        _sig("elo", 1.2, 0.9),
        _sig("weather", 0.93, 0.93, kind="global"),
    ]
    res = FactorEnsemble(aggregation="geom").combine(signals)
    # Tilt: nur elo → home-Mult exp(1·ln 1.2)=1.2, dann ×0.93.
    assert abs(res.lambda_home_multiplier - 1.2 * 0.93) < 1e-9
    assert res.breakdown_payload["lambda_aggregation"] == "geom"
