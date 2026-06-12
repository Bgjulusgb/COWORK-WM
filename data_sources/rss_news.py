"""RSS football-news connector — injury / availability mining.

Feeds (all free, no key, no account):
    BBC Sport     https://feeds.bbci.co.uk/sport/football/rss.xml
    Guardian      https://www.theguardian.com/football/rss
    ESPN Soccer   https://www.espn.com/espn/rss/soccer/news

Each entry is attributed to a team by matching the nation's name + aliases
(team_codes) in the title/summary, and scored for injury severity by keyword.
A spaCy-NER player→squad hook can refine attribution later; name matching keeps
v3 reliable without a roster table. `feedparser` is imported lazily so the
connector degrades to the deterministic mock when it (or the network) is absent.
"""
from __future__ import annotations

import asyncio

import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import rss_mock
from data_sources.schemas import InjuryNewsItem
from data_sources.team_codes import CODE_TO_NAMES, preferred_name, to_code

log = structlog.get_logger("data_sources.rss_news")

# Lazy, cached spaCy NER model (separate from the preprocessing pipeline, which
# runs with NER disabled). None when spaCy/the model isn't installed → the
# connector falls back to alias substring matching.
_NLP = None
_NLP_TRIED = False


def _get_nlp():
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy

        _NLP = spacy.load(
            "en_core_web_sm",
            disable=["lemmatizer", "tagger", "attribute_ruler", "parser"],
        )
    except Exception:
        _NLP = None
    return _NLP


def _entities(nlp, text: str) -> tuple[set[str], list[str]]:
    """(team codes from GPE/ORG/NORP entities, PERSON names) in a headline."""
    codes: set[str] = set()
    persons: list[str] = []
    try:
        doc = nlp(text)
    except Exception:
        return codes, persons
    return _extract_ents(doc)


def _extract_ents(doc) -> tuple[set[str], list[str]]:
    codes: set[str] = set()
    persons: list[str] = []
    for ent in getattr(doc, "ents", []) or []:
        if ent.label_ in ("GPE", "ORG", "NORP"):
            code = to_code(ent.text)
            if code:
                codes.add(code)
        elif ent.label_ == "PERSON":
            persons.append(ent.text.strip())
    return codes, persons


def _pipe_blocking(nlp, blobs: list[str]) -> list[tuple[set[str], list[str]]]:
    """Sync batch helper — runs nlp.pipe over *blobs* and extracts entities.
    Stays sync because spaCy itself is CPU-bound; the async wrapper
    :func:`_entities_batch` pushes it into a thread."""
    try:
        docs = list(nlp.pipe(blobs, batch_size=16))
    except Exception:
        return [(set(), []) for _ in blobs]
    return [_extract_ents(d) for d in docs]


async def _entities_batch(nlp, blobs: list[str]) -> list[tuple[set[str], list[str]]]:
    """Async wrapper around spaCy's batched pipe — runs the CPU-bound spaCy work
    in a worker thread so the asyncio event loop stays responsive (K4)."""
    if not blobs:
        return []
    return await asyncio.to_thread(_pipe_blocking, nlp, blobs)

_FEEDS = (
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.theguardian.com/football/rss",
    "https://www.espn.com/espn/rss/soccer/news",
)
_RSS_TTL_S = 30 * 60  # headlines churn; 30-minute cache

# (keywords, severity) — highest matching severity wins per entry.
_SEVERITY: tuple[tuple[tuple[str, ...], float], ...] = (
    (("ruled out", "out of the", "will miss", "sidelined", "season-ending",
      "acl", "ligament", "surgery", "tear"), 0.8),
    (("suspended", "ban ", "banned", "red card"), 0.7),
    (("doubt", "fitness test", "knock", "strain", "assessed", "scan"), 0.4),
    (("injury", "injured", "fitness"), 0.3),
)


def _severity(text: str) -> float:
    t = text.lower()
    best = 0.0
    for keywords, score in _SEVERITY:
        if any(k in t for k in keywords) and score > best:
            best = score
    return best


def _aliases(code: str) -> list[str]:
    names = list(CODE_TO_NAMES.get(code.upper(), [])) or [preferred_name(code)]
    # Only reasonably distinctive names (avoid 1-2 letter false positives).
    return [n.lower() for n in names if len(n) >= 4]


class RssNewsConnector(BaseConnector):
    connector_name = "rss_news"

    async def get_team_news(self, code: str, name: str | None = None) -> FetchResult:
        code = code.upper()
        if settings.use_mock_rss:
            return FetchResult(rss_mock.injury_news(code), "mock", None, "mock")

        try:
            import feedparser  # noqa: F401
        except Exception:
            log.debug("feedparser_missing_using_mock")
            return FetchResult(rss_mock.injury_news(code), "mock", None, "mock")

        entries = await self._all_entries()
        if entries is None:
            return FetchResult(rss_mock.injury_news(code), "mock", None, "mock")

        nlp = _get_nlp()
        aliases = _aliases(code)
        # K4: erst alle injury-relevanten Headlines sammeln, dann NER batched
        # via asyncio.to_thread laufen lassen — sonst blockt jeder spaCy-Call
        # den Event-Loop fuer 5–20 ms.
        candidates: list[tuple[str, str, str, str, str, float]] = []
        for title, summary, link, src in entries:
            blob = f"{title} {summary}"
            sev = _severity(blob)
            if sev <= 0.0:
                continue
            candidates.append((title, summary, link, src, blob, sev))

        ner_results: list[tuple[set[str], list[str]]]
        if nlp and candidates:
            ner_results = await _entities_batch(nlp, [c[4] for c in candidates])
        else:
            ner_results = [(set(), []) for _ in candidates]

        items: list[InjuryNewsItem] = []
        for (title, summary, link, src, blob, sev), (ner_codes, persons) in zip(
            candidates, ner_results
        ):
            matched_by_entity = code in ner_codes
            matched_by_alias = any(a in blob.lower() for a in aliases)
            if not (matched_by_entity or matched_by_alias):
                continue
            impact = min(1.0, sev + 0.1) if persons else sev
            items.append(InjuryNewsItem(
                source=src, team_code=code, headline=title[:200], impact=impact,
                url=link or None, player=persons[0] if persons else None,
            ))
        # No live injury hits is a valid answer (mode stays live), but if the
        # whole fetch produced nothing usable we already returned mock above.
        return FetchResult(items, "live", None, self.connector_name)

    async def _all_entries(self) -> list[tuple[str, str, str, str]] | None:
        import feedparser

        out: list[tuple[str, str, str, str]] = []
        any_ok = False
        for url in _FEEDS:
            res = await self._get_text(url, ttl_s=_RSS_TTL_S)
            if not res.ok or not isinstance(res.data, str):
                continue
            any_ok = True
            parsed = feedparser.parse(res.data)
            src = (parsed.feed or {}).get("title", url) if hasattr(parsed, "feed") else url
            for e in getattr(parsed, "entries", []) or []:
                out.append((
                    getattr(e, "title", "") or "",
                    getattr(e, "summary", "") or getattr(e, "description", "") or "",
                    getattr(e, "link", "") or "",
                    src,
                ))
        return out if any_ok else None


__all__ = ["RssNewsConnector"]
