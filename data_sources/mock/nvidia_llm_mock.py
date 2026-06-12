"""Deterministic NVIDIA-LLM aspect-sentiment mock.

Used when ``USE_NVIDIA_LLM=false`` or no API key is present. Returns a small
synthetic payload shaped like the production ``score_sentiment`` response so
the LlmSentimentFactor can be tested without network access.
"""
from __future__ import annotations

import hashlib


def _seed(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode("utf-8")).hexdigest(), 16)


def aspect_sentiment(home_code: str, away_code: str, sample_texts: int = 0) -> dict:
    """Synthetic LLM scoring payload for one match."""
    h = _seed("nvidia", home_code.upper(), away_code.upper())
    # Bias home polarity slightly up, away slightly down, with deterministic jitter.
    home_pol = round(0.08 + (h % 17) * 0.012, 3)
    away_pol = round(-0.05 + ((h >> 4) % 13) * 0.011, 3)
    return {
        "model": "mock-llm-aspect",
        "samples": max(sample_texts, 8),
        "home": {
            "polarity": home_pol,
            "intensity": round(0.55 + (h % 7) * 0.04, 3),
            "confidence": 0.7,
            "aspects": {"attack": 0.4, "defence": 0.0, "morale": 0.3},
        },
        "away": {
            "polarity": away_pol,
            "intensity": round(0.45 + ((h >> 8) % 9) * 0.03, 3),
            "confidence": 0.65,
            "aspects": {"attack": 0.1, "defence": -0.2, "morale": 0.0},
        },
    }


__all__ = ["aspect_sentiment"]
