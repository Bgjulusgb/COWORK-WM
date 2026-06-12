import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from preprocessing.sport_slang import expand_slang
from preprocessing.text_cleaner import clean_text

# Negations MUST survive stopword removal — they invert sentiment.
_NEGATIONS = {
    "not", "no", "never", "without", "nor", "neither", "none", "nothing", "nobody",
    "cannot", "isnt", "arent", "wasnt", "werent", "doesnt", "didnt", "dont",
    "wont", "wouldnt", "shouldnt", "couldnt", "hadnt", "hasnt", "havent",
}

_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "just", "as", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we", "us",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their",
}


def _build_keyword_pattern(keywords: List[str]) -> re.Pattern | None:
    """Compile keywords into a single word-boundary regex for accurate matching.

    Prevents substring false-positives like 'hon' matching 'honestly'
    or 'tri' matching 'triple'.
    """
    if not keywords:
        return None
    escaped = [re.escape(k.lower()) for k in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


@dataclass
class ProcessedText:
    original: str
    cleaned: str
    tokens: List[str]
    lemmas: List[str]
    team_attribution: str  # "home" | "away" | "neutral"
    engagement_weight: float = 0.0
    language: str = "en"


class PreprocessingPipeline:
    """Spec order: clean -> slang -> tokenize/lemmatize -> attribute -> weight.

    Negations are preserved through stopword removal so that VADER and friends
    can correctly invert polarity downstream.
    """

    def __init__(
        self,
        slang_dict: Optional[Dict[str, str]] = None,
        home_keywords: Optional[List[str]] = None,
        away_keywords: Optional[List[str]] = None,
        *,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.slang_dict = slang_dict or {}
        self._home_pattern = _build_keyword_pattern(home_keywords or [])
        self._away_pattern = _build_keyword_pattern(away_keywords or [])
        # Keep lowercased lists for backward compatibility
        self.home_keywords = [k.lower() for k in (home_keywords or [])]
        self.away_keywords = [k.lower() for k in (away_keywords or [])]
        self._nlp = None
        self._stopwords = _FALLBACK_STOPWORDS
        try:
            import spacy

            self._nlp = spacy.load(spacy_model, disable=["ner", "parser"])
            self._stopwords = self._nlp.Defaults.stop_words
        except Exception:
            # spaCy or model missing — fall back to regex tokenizer
            self._nlp = None

    def process(
        self,
        raw_text: str,
        *,
        score: int = 0,
        num_comments: int = 0,
        upvote_ratio: float = 1.0,
    ) -> ProcessedText:
        cleaned = clean_text(raw_text)
        cleaned = expand_slang(cleaned, self.slang_dict)

        if self._nlp is not None:
            doc = self._nlp(cleaned)
            tokens: List[str] = []
            lemmas: List[str] = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                t = tok.text
                lemma = tok.lemma_.lower() if tok.lemma_ else t.lower()
                if len(t) < 2 or len(t) > 50:
                    continue
                if t.lower() in _NEGATIONS or lemma in _NEGATIONS:
                    tokens.append(t.lower())
                    lemmas.append(lemma)
                    continue
                if tok.is_stop:
                    continue
                tokens.append(t.lower())
                lemmas.append(lemma)
        else:
            raw_tokens = [t for t in cleaned.split() if 2 <= len(t) <= 50]
            tokens = []
            lemmas = []
            for t in raw_tokens:
                if t in _NEGATIONS:
                    tokens.append(t)
                    lemmas.append(t)
                    continue
                if t in self._stopwords:
                    continue
                tokens.append(t)
                lemmas.append(t)

        attribution = self._attribute(cleaned)
        weight = self._engagement_weight(score, num_comments, upvote_ratio)

        return ProcessedText(
            original=raw_text,
            cleaned=cleaned,
            tokens=tokens,
            lemmas=lemmas,
            team_attribution=attribution,
            engagement_weight=weight,
        )

    def _attribute(self, text: str) -> str:
        has_home = bool(self._home_pattern and self._home_pattern.search(text))
        has_away = bool(self._away_pattern and self._away_pattern.search(text))
        if has_home and not has_away:
            return "home"
        if has_away and not has_home:
            return "away"
        return "neutral"

    @staticmethod
    def _engagement_weight(score: int, num_comments: int, upvote_ratio: float) -> float:
        import math

        return math.log1p(max(score, 0)) * math.log1p(max(num_comments, 0)) * max(min(upvote_ratio, 1.0), 0.0)
