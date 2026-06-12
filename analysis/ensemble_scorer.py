"""VADER + TextBlob + (optional) RoBERTa weighted sentiment ensemble.

Design notes
------------
- Weights are dynamic: short texts lean on VADER, long texts on RoBERTa.
- Non-English text (even after translation) downweights both English-trained
  models (VADER & RoBERTa) — TextBlob's polarity heuristic is more robust.
- A lightweight sarcasm heuristic (Reddit /s tag, ALL-CAPS positives in
  scare quotes) flips polarity when detected. Heuristic only — keep an eye on
  false positives, full irony model is a separate optional component.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Dict, Optional

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import settings


@dataclass
class SentimentResult:
    vader_score: float
    textblob_polarity: float
    textblob_subjectivity: float
    roberta_positive: Optional[float]
    roberta_neutral: Optional[float]
    roberta_negative: Optional[float]
    roberta_emotion: Optional[str]
    ensemble_score: float
    confidence: float
    sarcasm_detected: bool = False


# Sarcasm heuristics — conservative on purpose. A noisy classifier here would
# corrupt the entire downstream sentiment signal.
_SARCASM_TAG = re.compile(r"(?:\s|^)/s\b", re.IGNORECASE)
_SCARE_QUOTES_POSITIVE = re.compile(
    r"\"(amazing|great|brilliant|fantastic|world.?class|legend|genius)\"",
    re.IGNORECASE,
)
_POSITIVE_WORDS_FOR_CAPS = {
    "amazing", "brilliant", "fantastic", "great", "incredible",
    "legendary", "world-class", "worldclass", "genius",
}


def _detect_sarcasm(text: str) -> bool:
    if not text:
        return False
    if _SARCASM_TAG.search(text):
        return True
    if _SCARE_QUOTES_POSITIVE.search(text):
        return True
    # ALL-CAPS positive word in an otherwise mixed-case sentence (e.g.
    # "they're definitely going to WIN this one for sure").
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20:
        for word in re.findall(r"\b[A-Z]{4,}\b", text):
            if word.lower() in _POSITIVE_WORDS_FOR_CAPS:
                lower_letters = sum(1 for c in letters if c.islower())
                if lower_letters / len(letters) > 0.5:
                    return True
    return False


class SentimentEnsemble:
    """VADER + TextBlob (+ optional RoBERTa) weighted ensemble."""

    # Default static weights — only used when dynamic mode is off or as a base.
    _DEFAULT_WEIGHTS: Dict[str, float] = {"vader": 0.55, "textblob": 0.25, "roberta": 0.20}

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        custom_vader_lexicon: Optional[Dict[str, float]] = None,
        *,
        dynamic_weighting: bool = True,
    ) -> None:
        self._base_weights = {**self._DEFAULT_WEIGHTS, **(weights or {})}
        self.dynamic_weighting = dynamic_weighting
        self.vader = SentimentIntensityAnalyzer()
        if custom_vader_lexicon:
            self.vader.lexicon.update(custom_vader_lexicon)
        self._roberta = None
        if settings.use_roberta:
            from analysis.roberta_scorer import RoBERTaScorer
            self._roberta = RoBERTaScorer()

    def _length_adjusted_weights(self, text: str) -> Dict[str, float]:
        """Tilt weights toward VADER for short texts, RoBERTa for long ones.

        Empirically: VADER's lexicon dominates on tweets/short comments where
        RoBERTa has too little context; RoBERTa wins on multi-sentence rants
        where lexical sentiment is fragmented.
        """
        w = dict(self._base_weights)
        if not self.dynamic_weighting:
            return w
        word_count = len(text.split())
        if word_count < 12:
            w["vader"] *= 1.3
            w["roberta"] *= 0.6
        elif word_count > 60:
            w["vader"] *= 0.8
            w["roberta"] *= 1.4
        return w

    def score(self, text: str, source_language: str = "en") -> SentimentResult:
        if not text:
            return SentimentResult(0, 0, 0, None, None, None, None, 0, 0)

        sarcasm = _detect_sarcasm(text)

        vader = self.vader.polarity_scores(text)
        vader_score = vader["compound"]

        blob = TextBlob(text)
        tb_pol = float(blob.sentiment.polarity)
        tb_sub = float(blob.sentiment.subjectivity)

        roberta_polarity: Optional[float] = None
        roberta_emotion: Optional[str] = None
        roberta_pos: Optional[float] = None
        roberta_neu: Optional[float] = None
        roberta_neg: Optional[float] = None
        if self._roberta is not None:
            r = self._roberta.score_full(text)
            if r is not None:
                roberta_polarity = r.polarity
                roberta_emotion = r.emotion
                roberta_pos = r.positive
                roberta_neu = r.neutral
                roberta_neg = r.negative

        w = self._length_adjusted_weights(text)
        w_v = w["vader"]
        w_t = w["textblob"]
        w_r = w.get("roberta", 0.20)

        if source_language != "en":
            # BUG-04 fix: VADER *and* RoBERTa are English-trained — both get
            # demoted, not just VADER. TextBlob's polarity is closer to
            # language-agnostic and rises in relative weight.
            w_v *= 0.6
            w_r *= 0.5
            w_t *= 1.4

        if roberta_polarity is not None:
            denom = w_v + w_r + w_t
            ensemble = (w_v * vader_score + w_r * roberta_polarity + w_t * tb_pol) / denom
        else:
            denom = w_v + w_t
            ensemble = (w_v * vader_score + w_t * tb_pol) / denom

        # Sarcasm flips polarity but doesn't amplify magnitude. A sarcastic
        # "oh yeah, BRILLIANT" stays mildly negative rather than becoming
        # strongly negative — we don't trust the heuristic that much.
        if sarcasm and abs(ensemble) > 0.1:
            ensemble = -0.7 * ensemble

        scorer_values = [vader_score, tb_pol]
        if roberta_polarity is not None:
            scorer_values.append(roberta_polarity)
        std = statistics.stdev(scorer_values) if len(scorer_values) >= 2 else 0.0
        confidence = float(max(0.0, min(1.0, 1.0 - std)))

        return SentimentResult(
            vader_score=vader_score,
            textblob_polarity=tb_pol,
            textblob_subjectivity=tb_sub,
            # BUG-02 fix: store real class probabilities (sum-to-1), not
            # the signed-scalar split that was nonsensical before.
            roberta_positive=roberta_pos,
            roberta_neutral=roberta_neu,
            roberta_negative=roberta_neg,
            roberta_emotion=roberta_emotion,
            ensemble_score=float(max(-1.0, min(1.0, ensemble))),
            confidence=confidence,
            sarcasm_detected=sarcasm,
        )
