"""K4: spaCy NER darf den asyncio Event-Loop nicht blockieren.

Wir testen die neue `_entities_batch`-Helper (Batch via nlp.pipe + asyncio.to_thread):
* Funktional aequivalent zur sync _entities Variante
* Blockt den Event-Loop nicht — eine parallele asyncio-Task kann fortschreiten,
  selbst wenn die NER-Verarbeitung "langsam" ist.
"""
from __future__ import annotations

import asyncio
import time
from typing import Iterable

import pytest

from data_sources import rss_news


class _FakeEnt:
    def __init__(self, text: str, label_: str):
        self.text = text
        self.label_ = label_


class _FakeDoc:
    def __init__(self, ents: list[_FakeEnt]):
        self.ents = ents


class _SlowFakeNlp:
    """Simulates a spaCy model — each `nlp(text)` and each batch element in
    `pipe(...)` takes ``per_call_s`` seconds of *blocking* work."""

    def __init__(self, per_call_s: float = 0.03):
        self.per_call_s = per_call_s

    def __call__(self, text: str) -> _FakeDoc:
        time.sleep(self.per_call_s)
        return self._doc_for(text)

    def pipe(self, texts: Iterable[str], batch_size: int = 16):
        # Batched, but still blocking (spaCy stays CPU-bound).
        for t in texts:
            time.sleep(self.per_call_s)
            yield self._doc_for(t)

    @staticmethod
    def _doc_for(text: str) -> _FakeDoc:
        ents: list[_FakeEnt] = []
        if "Germany" in text:
            ents.append(_FakeEnt("Germany", "GPE"))
        if "Müller" in text or "Mueller" in text:
            ents.append(_FakeEnt("Müller", "PERSON"))
        return _FakeDoc(ents)


@pytest.mark.asyncio
async def test_entities_batch_returns_same_as_sync(monkeypatch):
    """Batch-async-Variante muss funktional aequivalent zur Schleife sein."""
    nlp = _SlowFakeNlp(per_call_s=0.0)
    monkeypatch.setattr(rss_news, "to_code", lambda s: "DEU" if s == "Germany" else None)

    blobs = [
        "Germany squad has a knock — Müller doubt",
        "France ruled out — no clue",
        "Germany Mueller fitness test",
    ]
    sync_results = [rss_news._entities(nlp, b) for b in blobs]
    batched = await rss_news._entities_batch(nlp, blobs)
    assert batched == sync_results


@pytest.mark.asyncio
async def test_entities_batch_does_not_block_event_loop(monkeypatch):
    """Während NER 200 ms blockiert (5 × 40 ms), muss eine parallele
    asyncio.sleep(0.05) prompt zurueckkommen — nicht erst nach NER-Ende."""
    nlp = _SlowFakeNlp(per_call_s=0.04)
    monkeypatch.setattr(rss_news, "to_code", lambda s: None)

    blobs = [f"text {i}" for i in range(5)]

    parallel_done_at: list[float] = []

    async def parallel_marker():
        await asyncio.sleep(0.05)
        parallel_done_at.append(time.monotonic())

    start = time.monotonic()
    marker_task = asyncio.create_task(parallel_marker())
    await rss_news._entities_batch(nlp, blobs)
    ner_done_at = time.monotonic()
    await marker_task

    assert parallel_done_at, "parallel marker never completed"
    marker_elapsed = parallel_done_at[0] - start
    ner_elapsed = ner_done_at - start
    # NER will take ~5 × 40 ms = 200 ms total blocking work.
    # marker should complete around 50 ms — well before NER finishes.
    assert marker_elapsed < ner_elapsed - 0.05, (
        f"parallel marker waited {marker_elapsed:.3f}s but NER took {ner_elapsed:.3f}s — "
        f"that means NER blocked the event loop"
    )
