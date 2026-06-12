"""Slang expansion with module-level regex cache.

`expand_slang` used to compile a fresh regex for every term in every post
(BUG-10) — 20 terms x 200 posts = 4 000 compilations per crawl. We now cache
the compiled patterns keyed by the slang-dict identity so each unique dict
is built exactly once.
"""
from __future__ import annotations

import re
import threading
from typing import Dict, List, Tuple


# WeakValueDictionary won't help here because the entries are tuples of
# compiled patterns. Use an LRU-ish cap to stop unbounded growth if callers
# pass many distinct dicts.
_CACHE: dict[Tuple[Tuple[str, str], ...], List[Tuple[re.Pattern, str]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_CAP = 64


def _compile(slang_dict: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    """Return `(compiled_pattern, replacement)` list for this slang dict.

    Patterns are sorted by descending term length so multi-word slang
    ("park the bus") is replaced before single-word substrings.
    """
    key = tuple(sorted((k.lower(), v.lower()) for k, v in slang_dict.items()))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    items = sorted(slang_dict.items(), key=lambda kv: -len(kv[0]))
    compiled = [
        (re.compile(r"\b" + re.escape(term.lower()) + r"\b", re.IGNORECASE), replacement.lower())
        for term, replacement in items
    ]
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_CAP:
            # Drop an arbitrary half — simple and bounded; the cache is keyed
            # by per-match slang dicts, hits are very common in practice.
            for k in list(_CACHE.keys())[: _CACHE_CAP // 2]:
                _CACHE.pop(k, None)
        _CACHE[key] = compiled
    return compiled


def expand_slang(text: str, slang_dict: Dict[str, str]) -> str:
    if not text or not slang_dict:
        return text
    for pattern, replacement in _compile(slang_dict):
        text = pattern.sub(replacement, text)
    return text
