"""Lightweight bot / low-quality content heuristics.

Two stages:

1. `is_bot_author(name)` — flags accounts that match the common bot naming
   patterns Reddit suffers from (RemindMe!, autotldr, b0tname, etc.).
2. `is_low_quality(text)` — flags posts that are templated, all-caps shouting,
   too short to score meaningfully, or look like cross-posted promo spam.

The thresholds are deliberately conservative — we want to drop obvious junk
without throwing out a passionate fan typing in caps.
"""
from __future__ import annotations

import re
from typing import Optional


_BOT_NAME_PATTERNS = [
    re.compile(r".*bot$", re.IGNORECASE),
    re.compile(r"^bot[_\-]?", re.IGNORECASE),
    # BUG-07 fix: was `^auto[_\-]?` which captured "autograph_fan",
    # "autonomous_driver" etc. Restrict to known bot stems.
    re.compile(r"^auto[_\-]?(mod|moderator|tldr|reply|bot|remove|post)", re.IGNORECASE),
    re.compile(r"^repostsleuthbot$", re.IGNORECASE),
    re.compile(r"^remindmebot$", re.IGNORECASE),
    re.compile(r"^autotldr$", re.IGNORECASE),
    re.compile(r"^automoderator$", re.IGNORECASE),
    re.compile(r"^converter[_\-]?bot$", re.IGNORECASE),
    re.compile(r"^sneakpeekbot$", re.IGNORECASE),
    re.compile(r"^sub.*help.*bot$", re.IGNORECASE),
]

_PROMO_PATTERNS = [
    re.compile(r"\b(check out|visit|sign up|claim your|bonus code|free bet)\b", re.IGNORECASE),
    re.compile(r"https?://\S*(bet|odds|casino|crypto)", re.IGNORECASE),
    re.compile(r"\b(use code|promo code|referral)\b", re.IGNORECASE),
]

_BOT_FOOTPRINTS = (
    "i am a bot",
    "this action was performed automatically",
    "beep boop",
    "contact the moderators of this subreddit",
)

# BUG-11: deleted/removed accounts have no traceable history and are typically
# either bot-purged spam or off-platform context-stripped — drop them.
_GHOST_AUTHORS = {
    "[deleted]", "[removed]", "deleted", "removed",
    "automoderator", "automod",
}


def is_bot_author(author: Optional[str]) -> bool:
    if not author:
        return False
    name = author.strip()
    if name.lower() in _GHOST_AUTHORS:
        return True
    if name == "unknown":
        # Crawler placeholder when the API didn't return an author at all.
        return True
    return any(p.match(name) for p in _BOT_NAME_PATTERNS)


def is_low_quality(text: Optional[str], *, min_chars: int = 8) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < min_chars:
        return True

    low = t.lower()
    if any(fp in low for fp in _BOT_FOOTPRINTS):
        return True

    if any(p.search(t) for p in _PROMO_PATTERNS):
        return True

    # ALL CAPS rage detection — only if long enough to matter
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 20:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.9:
            return True

    # Word salad: very high punctuation density
    non_space = [c for c in t if not c.isspace()]
    if non_space:
        punct_ratio = sum(1 for c in non_space if not (c.isalnum())) / len(non_space)
        if punct_ratio > 0.6:
            return True

    return False


def should_filter(author: Optional[str], text: Optional[str]) -> bool:
    return is_bot_author(author) or is_low_quality(text)
