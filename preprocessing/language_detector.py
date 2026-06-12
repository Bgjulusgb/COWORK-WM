"""Language detection + caching translation layer.

IMPROVE-10:
  - sha256 cache keyed by (source_lang, text). Persisted in `translation_cache`.
  - langdetect is unreliable below ~50 chars — short texts default to English
    to avoid spurious translation churn.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Optional

import structlog
from langdetect import DetectorFactory, detect
from deep_translator import GoogleTranslator

log = structlog.get_logger("preprocessing.language_detector")

DetectorFactory.seed = 0

_MIN_LANGDETECT_CHARS = 50
_MIN_TRANSLATE_CHARS = 10

# In-process LRU-ish cache to avoid hitting DB on every call.
_MEMORY_CACHE: dict[str, str] = {}
_MEMORY_CAP = 4096


def detect_language(text: str) -> str:
    if not text:
        return "en"
    stripped = text.strip()
    if len(stripped) < _MIN_LANGDETECT_CHARS:
        # langdetect is famously noisy on short text → treat as English to
        # skip translation work that would degrade rather than help.
        return "en"
    try:
        return detect(stripped)
    except Exception:
        return "en"


def _cache_key(source_lang: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_lang}\x00{text}".encode("utf-8")).hexdigest()
    return digest


def _remember(key: str, value: str) -> None:
    if len(_MEMORY_CACHE) >= _MEMORY_CAP:
        # Drop the oldest ~25% of entries — cheap and bounded.
        drop = list(_MEMORY_CACHE.keys())[: _MEMORY_CAP // 4]
        for k in drop:
            _MEMORY_CACHE.pop(k, None)
    _MEMORY_CACHE[key] = value


def _cache_get(key: str) -> Optional[str]:
    return _MEMORY_CACHE.get(key)


async def _db_cache_get(key: str) -> Optional[str]:
    """Persistent cache lookup. Returns None on any error."""
    try:
        from db.database import AsyncSessionLocal
        from db.models import TranslationCache
        async with AsyncSessionLocal() as session:
            row = await session.get(TranslationCache, key)
            return row.translated if row else None
    except Exception:
        return None


async def _db_cache_set(key: str, source_lang: str, translated: str) -> None:
    try:
        from db.database import AsyncSessionLocal
        from db.models import TranslationCache
        async with AsyncSessionLocal() as session:
            existing = await session.get(TranslationCache, key)
            if existing is not None:
                return
            session.add(
                TranslationCache(
                    id=key,
                    source_lang=source_lang,
                    text_hash_prefix=key[:16],
                    translated=translated,
                    created_at=datetime.utcnow(),
                )
            )
            await session.commit()
    except Exception as exc:
        log.debug("translation_cache_write_failed", error=str(exc))


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en" or len(text.strip()) < _MIN_TRANSLATE_CHARS:
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception as exc:
        log.warning("translation_failed", source_lang=source_lang, error=str(exc))
        return text


async def translate_to_english_async(text: str, source_lang: str) -> str:
    if source_lang == "en" or len(text.strip()) < _MIN_TRANSLATE_CHARS:
        return text

    key = _cache_key(source_lang, text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    persisted = await _db_cache_get(key)
    if persisted is not None:
        _remember(key, persisted)
        return persisted

    result = await asyncio.to_thread(translate_to_english, text, source_lang)
    if result and result != text:
        _remember(key, result)
        await _db_cache_set(key, source_lang, result)
    return result


async def translate_batch_async(items: list[tuple[str, str]]) -> list[str]:
    tasks = [translate_to_english_async(text, lang) for text, lang in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[str] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.warning("batch_translation_failed", index=i, error=str(r))
            out.append(items[i][0])
        else:
            out.append(r)
    return out
