"""FIFA 3-letter code ⇄ team-name mapping for the 48 WC-2026 teams.

External sources (openfootball, TheSportsDB) identify teams by full English
name, sometimes with several spellings ("South Korea" vs "Korea Republic",
"IR Iran" vs "Iran"). The factors and the YAML configs work in FIFA codes.
This module bridges the two with an accent-insensitive lookup.

Canonical direction is CODE_TO_NAMES (one code → its known aliases). NAME_TO_CODE
is derived from it on import, so adding an alias is a one-line edit.
"""
from __future__ import annotations

import unicodedata

# FIFA code → list of name spellings seen across the free sources. The first
# entry is the "preferred" display name; the rest are aliases for matching.
CODE_TO_NAMES: dict[str, list[str]] = {
    # Group A
    "MEX": ["Mexico", "México"],
    "RSA": ["South Africa"],
    "KOR": ["South Korea", "Korea Republic", "Korea, South", "Republic of Korea"],
    "CZE": ["Czech Republic", "Czechia"],
    # Group B
    "CAN": ["Canada"],
    "BIH": ["Bosnia & Herzegovina", "Bosnia and Herzegovina", "Bosnia-Herzegovina"],
    "QAT": ["Qatar"],
    "SUI": ["Switzerland", "Schweiz"],
    # Group C
    "BRA": ["Brazil", "Brasil"],
    "MAR": ["Morocco", "Maroc"],
    "HAI": ["Haiti", "Haïti"],
    "SCO": ["Scotland"],
    # Group D
    "USA": ["USA", "United States", "United States of America"],
    "PAR": ["Paraguay"],
    "AUS": ["Australia"],
    "TUR": ["Turkey", "Türkiye", "Turkiye"],
    # Group E
    "GER": ["Germany", "Deutschland"],
    "CUW": ["Curaçao", "Curacao"],
    "CIV": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire", "Cote d Ivoire"],
    "ECU": ["Ecuador"],
    # Group F
    "NED": ["Netherlands", "Holland"],
    "JPN": ["Japan"],
    "SWE": ["Sweden"],
    "TUN": ["Tunisia"],
    # Group G
    "BEL": ["Belgium", "Belgique"],
    "EGY": ["Egypt"],
    "IRN": ["Iran", "IR Iran", "Iran, Islamic Republic of"],
    "NZL": ["New Zealand"],
    # Group H
    "ESP": ["Spain", "España", "Espana"],
    "CPV": ["Cape Verde", "Cabo Verde"],
    "KSA": ["Saudi Arabia"],
    "URU": ["Uruguay"],
    # Group I
    "FRA": ["France"],
    "SEN": ["Senegal", "Sénégal"],
    "IRQ": ["Iraq"],
    "NOR": ["Norway", "Norge"],
    # Group J
    "ARG": ["Argentina"],
    "ALG": ["Algeria", "Algérie"],
    "AUT": ["Austria", "Österreich", "Osterreich"],
    "JOR": ["Jordan"],
    # Group K
    "POR": ["Portugal"],
    "COD": ["DR Congo", "Congo DR", "Democratic Republic of the Congo", "Congo Kinshasa"],
    "UZB": ["Uzbekistan"],
    "COL": ["Colombia"],
    # Group L
    "ENG": ["England"],
    "CRO": ["Croatia", "Hrvatska"],
    "GHA": ["Ghana"],
    "PAN": ["Panama", "Panamá"],
}


def _normalise(name: str) -> str:
    """Lower-case, strip accents and punctuation so 'Côte d'Ivoire' and
    'cote divoire' collapse to the same key."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = "".join(c if c.isalnum() else " " for c in ascii_only.lower())
    return " ".join(cleaned.split())


# Derived reverse index: normalised name → FIFA code.
NAME_TO_CODE: dict[str, str] = {}
for _code, _names in CODE_TO_NAMES.items():
    NAME_TO_CODE[_normalise(_code)] = _code  # the code itself is a valid key
    for _n in _names:
        NAME_TO_CODE[_normalise(_n)] = _code


def to_code(name: str | None) -> str | None:
    """Resolve a team name (any known spelling) or a FIFA code to its code.
    Returns None when nothing matches — callers fall back to YAML data."""
    if not name:
        return None
    return NAME_TO_CODE.get(_normalise(name))


def preferred_name(code: str) -> str:
    """Display name for a code; the code itself if unknown."""
    names = CODE_TO_NAMES.get(code.upper())
    return names[0] if names else code.upper()


__all__ = ["CODE_TO_NAMES", "NAME_TO_CODE", "to_code", "preferred_name"]
