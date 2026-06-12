"""RoBERTa emotion scorer — lazy-loaded, optional (USE_ROBERTA=true in .env).

Model: j-hartmann/emotion-english-distilroberta-base
  ~85 MB, 7 emotions: joy, sadness, anger, fear, surprise, disgust, neutral
  Returns full class-probability distribution (top_k=None) and derives:
    - scalar polarity in [-1, 1] for ensemble blending
    - positive / neutral / negative buckets in [0, 1] for storage
    - dominant emotion label
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from transformers import Pipeline

# Positive emotions mapped to [0..1], negative to [-1..0]
_EMOTION_POLARITY: Dict[str, float] = {
    "joy": 1.0,
    "surprise": 0.4,
    "neutral": 0.0,
    "fear": -0.5,
    "disgust": -0.7,
    "sadness": -0.8,
    "anger": -0.9,
}

# Buckets for storing as positive/neutral/negative class-probabilities.
_POSITIVE_LABELS = {"joy", "surprise"}
_NEUTRAL_LABELS = {"neutral"}
_NEGATIVE_LABELS = {"anger", "sadness", "fear", "disgust"}

_DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def _model_name() -> str:
    try:
        from config.settings import settings
        return settings.roberta_model or _DEFAULT_MODEL
    except Exception:
        return _DEFAULT_MODEL


# Twitter-trained sentiment model (cardiffnlp/...) uses LABEL_0/LABEL_1/LABEL_2
# for negative/neutral/positive. The emotion model uses the 7 labels in
# _EMOTION_POLARITY. We support both transparently in score_full.
_CARDIFF_LABELS = {
    "LABEL_0": -1.0,  # negative
    "LABEL_1": 0.0,   # neutral
    "LABEL_2": 1.0,   # positive
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}


class RobertaResult:
    __slots__ = ("polarity", "positive", "neutral", "negative", "emotion", "scores")

    def __init__(
        self,
        polarity: float,
        positive: float,
        neutral: float,
        negative: float,
        emotion: str,
        scores: Dict[str, float],
    ) -> None:
        self.polarity = polarity
        self.positive = positive
        self.neutral = neutral
        self.negative = negative
        self.emotion = emotion
        self.scores = scores


class RoBERTaScorer:
    """Lazy-loaded emotion classifier.

    First call to .score() triggers model download (~85 MB). Subsequent calls
    use the cached in-memory pipeline.
    """

    def __init__(self) -> None:
        self._pipe: "Pipeline | None" = None
        self._available = True  # set False on load failure

    def _load(self) -> None:
        if self._pipe is not None or not self._available:
            return
        try:
            from transformers import pipeline  # type: ignore[import]
            # top_k=None returns the full probability distribution.
            self._pipe = pipeline(
                "text-classification",
                model=_model_name(),
                top_k=None,
                truncation=True,
                max_length=512,
            )
        except Exception as exc:
            self._available = False
            raise RuntimeError(
                f"RoBERTa load failed: {exc}. "
                "Install with: pip install transformers torch"
            ) from exc

    def score_full(self, text: str) -> RobertaResult | None:
        """Return all 7 class probabilities + derived polarity, or None if unavailable."""
        if not text or not self._available:
            return None
        try:
            self._load()
            raw = self._pipe(text[:512])  # type: ignore[index]
        except Exception:
            return None

        # Pipeline with top_k=None returns: [[{"label": ..., "score": ...}, ...]]
        inner = raw[0] if (raw and isinstance(raw, list) and raw and isinstance(raw[0], list)) else raw
        scores: Dict[str, float] = {item["label"]: float(item["score"]) for item in inner}

        # Cardiff sentiment model (3 labels) vs Hartmann emotion model (7 labels)
        # — auto-detect by overlap with known label sets.
        if any(lbl in _CARDIFF_LABELS for lbl in scores):
            polarity = sum(_CARDIFF_LABELS.get(lbl, 0.0) * p for lbl, p in scores.items())
            positive = sum(p for lbl, p in scores.items() if _CARDIFF_LABELS.get(lbl, 0.0) > 0)
            neutral = sum(p for lbl, p in scores.items() if _CARDIFF_LABELS.get(lbl, 0.0) == 0)
            negative = sum(p for lbl, p in scores.items() if _CARDIFF_LABELS.get(lbl, 0.0) < 0)
        else:
            polarity = sum(_EMOTION_POLARITY.get(lbl, 0.0) * p for lbl, p in scores.items())
            positive = sum(p for lbl, p in scores.items() if lbl in _POSITIVE_LABELS)
            neutral = sum(p for lbl, p in scores.items() if lbl in _NEUTRAL_LABELS)
            negative = sum(p for lbl, p in scores.items() if lbl in _NEGATIVE_LABELS)
        emotion = max(scores.items(), key=lambda kv: kv[1])[0]

        return RobertaResult(
            polarity=max(-1.0, min(1.0, polarity)),
            positive=positive,
            neutral=neutral,
            negative=negative,
            emotion=emotion,
            scores=scores,
        )

    def score(self, text: str) -> Tuple[float, str]:
        """Legacy 2-tuple form for callers that only want (scalar, emotion)."""
        r = self.score_full(text)
        if r is None:
            return 0.0, "neutral"
        return r.polarity, r.emotion

    @property
    def is_available(self) -> bool:
        return self._available and self._pipe is not None
