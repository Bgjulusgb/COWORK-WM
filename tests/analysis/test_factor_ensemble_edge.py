"""M1: factor_ensemble darf nicht crashen, wenn ein Faktor extrem kleine oder
NaN-aehnliche away_strength meldet. Pydantic clamped auf [0.3, 2.5], aber wir
testen den Pfad fuer den Fall, dass das Clamp unterschritten ist (z.B. via
direkter Manipulation vor dem Validator)."""
from __future__ import annotations

import math

from analysis.factor_ensemble import FactorEnsemble
from factors.base import FactorSignal


def _signal(name: str, home: float, away: float, weight: float = 0.3):
    return FactorSignal(
        name=name,
        home_strength=home,
        away_strength=away,
        weight=weight,
        confidence=0.8,
        available=True,
        source="test",
    )


def test_ensemble_handles_min_clamp_away_strength_safely():
    # Pydantic clamps to 0.3, so direct min away_strength = 0.3. The ratio
    # 2.5/0.3 ≈ 8.33 must not break statistics.stdev or downstream confidence.
    signals = [
        _signal("a", home=2.5, away=0.3),
        _signal("b", home=2.4, away=0.3),
    ]
    result = FactorEnsemble().combine(signals)
    assert math.isfinite(result.lambda_home_multiplier)
    assert math.isfinite(result.lambda_away_multiplier)
    assert 0.0 <= result.confidence <= 1.0


def test_ensemble_survives_post_validator_nan_via_object_mutation(monkeypatch):
    """If a downstream consumer mutates a signal to NaN after Pydantic validation
    (e.g. a faulty cache), the ensemble must not propagate NaN."""
    signals = [_signal("a", home=1.2, away=0.8), _signal("b", home=1.0, away=1.0)]
    # Force a NaN away_strength on the first signal by bypassing Pydantic.
    object.__setattr__(signals[0], "away_strength", float("nan"))
    result = FactorEnsemble().combine(signals)
    assert math.isfinite(result.lambda_home_multiplier)
    assert math.isfinite(result.lambda_away_multiplier)
    assert math.isfinite(result.confidence)
