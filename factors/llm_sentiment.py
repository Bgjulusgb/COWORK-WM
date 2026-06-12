"""NVIDIA-LLM aspect-sentiment factor.

Sits in the ensemble alongside :class:`SentimentFactor` (VADER/TextBlob/RoBERTa).
The LLM is consulted only when ``settings.use_nvidia_llm`` is on; the scorer
fills ``ctx.sentiment_payload["llm"]`` with::

    {"home": {"polarity", "intensity", "confidence", "aspects"}, "away": {...}}

Projection from polarity → strength mirrors the rule used by the classical
SentimentFactor so the two scales line up — the LLM signal then acts as a
peer voice in the ensemble, not a parallel scale. Confidence is the LLM's
self-reported confidence shrunk by the lower of the two aspect-vector norms
(disagreement between attack/defence/morale ⇒ less trustworthy).
"""
from __future__ import annotations

from factors.base import Factor, FactorContext, FactorSignal

_MAX_NUDGE = 0.12     # slightly higher cap than the classical scorer because
                       # LLM polarity is calibrated, not VADER-noisy.


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _aspect_norm(aspects: dict | None) -> float:
    if not isinstance(aspects, dict):
        return 0.0
    keys = ("attack", "defence", "morale")
    vals = [abs(float(aspects.get(k, 0.0) or 0.0)) for k in keys]
    return sum(vals) / len(keys)


class LlmSentimentFactor(Factor):
    name = "llm_sentiment"
    default_weight = 0.0     # off until USE_NVIDIA_LLM=true + key + a non-zero weight

    async def compute(self, ctx: FactorContext) -> FactorSignal:
        payload = (ctx.sentiment_payload or {}).get("llm")
        if not isinstance(payload, dict):
            return self._neutral(source="nvidia_llm", reason="no_payload")

        home = payload.get("home") or {}
        away = payload.get("away") or {}
        if not home and not away:
            return self._neutral(source="nvidia_llm", reason="empty_payload")

        home_pol = _clip(float(home.get("polarity", 0.0) or 0.0), -1.0, 1.0)
        away_pol = _clip(float(away.get("polarity", 0.0) or 0.0), -1.0, 1.0)
        home_int = _clip(float(home.get("intensity", 0.5) or 0.5), 0.0, 1.0)
        away_int = _clip(float(away.get("intensity", 0.5) or 0.5), 0.0, 1.0)

        # Intensity gates how much polarity matters — a high polarity but low
        # intensity post is a token "go team" rather than an actual mood shift.
        home_strength = 1.0 + _MAX_NUDGE * home_pol * (0.5 + 0.5 * home_int)
        away_strength = 1.0 + _MAX_NUDGE * away_pol * (0.5 + 0.5 * away_int)

        # Confidence = self-reported × aspect-vector-strength (disagreement damp).
        home_conf_raw = _clip(float(home.get("confidence", 0.5) or 0.5), 0.0, 1.0)
        away_conf_raw = _clip(float(away.get("confidence", 0.5) or 0.5), 0.0, 1.0)
        aspect_strength = (_aspect_norm(home.get("aspects")) + _aspect_norm(away.get("aspects"))) / 2.0
        confidence = _clip(0.5 * (home_conf_raw + away_conf_raw) * (0.6 + 0.4 * aspect_strength),
                           0.0, 1.0)

        source = payload.get("model") or "nvidia_llm"
        # The orchestrator records nvidia_llm provenance separately; we use the
        # factor-local model name as a more descriptive source label.
        return FactorSignal(
            name=self.name,
            home_strength=home_strength,
            away_strength=away_strength,
            weight=self.weight,
            confidence=confidence,
            available=True,
            source=source,
            raw_data={
                "home_polarity": round(home_pol, 4),
                "away_polarity": round(away_pol, 4),
                "home_intensity": round(home_int, 3),
                "away_intensity": round(away_int, 3),
                "samples": int(payload.get("samples", 0) or 0),
                "home_aspects": home.get("aspects"),
                "away_aspects": away.get("aspects"),
            },
        )


__all__ = ["LlmSentimentFactor"]
