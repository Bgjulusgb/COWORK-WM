"""Tests for analysis.factor_ensemble.FactorEnsemble."""
import pytest

from analysis.factor_ensemble import FactorEnsemble
from factors.base import FactorSignal


def _sig(name: str, *, home=1.0, away=1.0, weight=0.2, confidence=0.8, available=True):
    return FactorSignal(
        name=name,
        home_strength=home,
        away_strength=away,
        weight=weight,
        confidence=confidence,
        available=available,
        source="test",
    )


def test_single_factor_passes_through_unchanged():
    result = FactorEnsemble().combine([_sig("elo", home=1.2, away=0.8, weight=0.3)])
    assert result.lambda_home_multiplier == pytest.approx(1.2)
    assert result.lambda_away_multiplier == pytest.approx(0.8)


def test_two_factors_average_with_weights():
    sigs = [
        _sig("elo", home=1.4, away=0.6, weight=0.4, confidence=0.9),
        _sig("form", home=1.0, away=1.0, weight=0.2, confidence=0.7),
    ]
    result = FactorEnsemble().combine(sigs)
    # Effective weights: elo=0.667, form=0.333
    expected_home = (0.4 / 0.6) * 1.4 + (0.2 / 0.6) * 1.0
    expected_away = (0.4 / 0.6) * 0.6 + (0.2 / 0.6) * 1.0
    assert result.lambda_home_multiplier == pytest.approx(expected_home)
    assert result.lambda_away_multiplier == pytest.approx(expected_away)


def test_unavailable_factor_is_renormalised_out():
    sigs = [
        _sig("elo", home=1.4, away=0.6, weight=0.4),
        _sig("form", home=1.0, away=1.0, weight=0.2, available=False),
    ]
    result = FactorEnsemble().combine(sigs)
    # Only elo counts. Effective weight should be 1.0, signal passes through.
    assert result.lambda_home_multiplier == pytest.approx(1.4)
    assert result.lambda_away_multiplier == pytest.approx(0.6)
    elo_payload = next(s for s in result.breakdown_payload["signals"] if s["name"] == "elo")
    form_payload = next(s for s in result.breakdown_payload["signals"] if s["name"] == "form")
    assert elo_payload["effective_weight"] == pytest.approx(1.0)
    assert form_payload["effective_weight"] == pytest.approx(0.0)


def test_all_unavailable_yields_neutral_multipliers():
    sigs = [
        _sig("elo", available=False, weight=0.4),
        _sig("form", available=False, weight=0.2),
    ]
    result = FactorEnsemble().combine(sigs)
    assert result.lambda_home_multiplier == 1.0
    assert result.lambda_away_multiplier == 1.0
    assert result.confidence == 0.0
    assert any("No available factors" in n for n in result.breakdown_payload["notes"])


def test_confidence_combines_avg_and_agreement():
    # Two factors that agree → high confidence; both confidence 0.8 → ~0.8.
    sigs = [
        _sig("elo", home=1.2, away=0.8, weight=0.4, confidence=0.8),
        _sig("form", home=1.2, away=0.8, weight=0.3, confidence=0.8),
    ]
    result_agree = FactorEnsemble().combine(sigs)

    # Two factors that disagree → agreement penalty.
    sigs_disagree = [
        _sig("elo", home=1.5, away=0.5, weight=0.4, confidence=0.8),
        _sig("form", home=0.5, away=1.5, weight=0.3, confidence=0.8),
    ]
    result_disagree = FactorEnsemble().combine(sigs_disagree)

    assert result_agree.confidence > result_disagree.confidence


def test_breakdown_payload_includes_xg_proxy_flag():
    s = FactorSignal(
        name="goals",
        home_strength=1.1,
        away_strength=0.9,
        weight=0.15,
        confidence=0.7,
        available=True,
        source="goals_proxy",
        raw_data={"is_xg_proxy": True, "games": 10},
    )
    result = FactorEnsemble().combine([s])
    payload = result.breakdown_payload["signals"][0]
    assert payload["is_xg_proxy"] is True
    assert payload["raw_data"]["games"] == 10


def test_zero_weight_factor_is_skipped():
    sigs = [
        _sig("elo", home=1.4, away=0.6, weight=0.4),
        _sig("squad", home=1.5, away=0.5, weight=0.0, available=True),
    ]
    result = FactorEnsemble().combine(sigs)
    # squad has weight 0 → must not contribute.
    assert result.lambda_home_multiplier == pytest.approx(1.4)
    assert result.lambda_away_multiplier == pytest.approx(0.6)


def _global(name, *, damp, weight=0.05):
    s = FactorSignal(name=name, home_strength=damp, away_strength=damp, weight=weight,
                     confidence=0.6, available=True, source="env", kind="global")
    return s


def test_global_modifier_multiplies_not_averaged():
    # One tilt factor (elo) + one global damp (weather 0.90). The damp must
    # multiply the λ, not get averaged toward 1.0.
    sigs = [_sig("elo", home=1.2, away=0.8, weight=0.3), _global("weather", damp=0.90)]
    result = FactorEnsemble().combine(sigs)
    assert result.lambda_home_multiplier == pytest.approx(1.2 * 0.90)
    assert result.lambda_away_multiplier == pytest.approx(0.8 * 0.90)
    assert result.breakdown_payload["global_multiplier"] == pytest.approx(0.90)


def test_stacked_global_modifiers_are_floored():
    # Weather 0.85 × altitude 0.80 = 0.68, but the floor caps it at 0.82.
    sigs = [_sig("elo", home=1.0, away=1.0, weight=0.3),
            _global("weather", damp=0.85), _global("altitude", damp=0.80)]
    result = FactorEnsemble().combine(sigs)
    assert result.lambda_home_multiplier == pytest.approx(0.82)


def test_only_global_factor_starts_neutral():
    result = FactorEnsemble().combine([_global("altitude", damp=0.90)])
    # No tilt factor → base 1.0, damped by the global modifier.
    assert result.lambda_home_multiplier == pytest.approx(0.90)
    assert result.lambda_away_multiplier == pytest.approx(0.90)
