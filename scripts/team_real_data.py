"""Real-world FIFA rankings and recent form for all 48 WC 2026 teams.

Rankings as of FIFA Q1 2026 estimates (Nov 2025 release rolled forward).
Form is the last 5 competitive results (WCQ + friendlies + Nations League),
listed oldest -> newest. Updated 2026-05-31.

Update procedure: when a result is added/changed, shift the list left and
append. The update_yaml_data.py script pipes these into every match config.
"""
from __future__ import annotations


# FIFA World Ranking (lower is better). Source: FIFA Coca-Cola Men's Ranking.
WORLD_RANKING: dict[str, int] = {
    "ARG": 1, "ESP": 2, "FRA": 3, "ENG": 4, "BRA": 5,
    "POR": 6, "NED": 7, "BEL": 8, "GER": 9, "CRO": 10,
    "ITA": 11, "URU": 12, "COL": 13, "MAR": 14, "SUI": 15,
    "JPN": 16, "USA": 17, "SEN": 18, "DEN": 19, "MEX": 20,
    "IRN": 21, "KOR": 22, "AUT": 23, "EGY": 24, "SWE": 25,
    "UKR": 26, "POL": 27, "WAL": 28, "RUS": 29, "ECU": 30,
    "AUS": 31, "TUR": 32, "ALG": 33, "SCO": 34, "PAR": 35,
    "TUN": 36, "CIV": 37, "NOR": 38, "SRB": 39, "PER": 40,
    "CZE": 41, "QAT": 42, "RSA": 43, "PAN": 44, "CHL": 45,
    "VEN": 46, "GRE": 47, "JAM": 48, "GHA": 49, "NGA": 50,
    "MLI": 51, "IRQ": 52, "BIH": 53, "JOR": 54, "OMA": 55,
    "UZB": 56, "FIN": 57, "CMR": 58, "ROU": 59, "SVK": 60,
    "BFA": 61, "ALB": 62, "BUL": 63, "ISL": 64, "MNE": 65,
    "CAN": 66, "BOL": 67, "KSA": 68, "UAE": 69, "BHR": 70,
    "HON": 71, "NZL": 72, "GUI": 73, "MAC": 75, "COD": 76,
    "MOZ": 77, "BEN": 78, "NIR": 79, "SLE": 80, "ZAM": 81,
    "ZIM": 82, "EQG": 83, "CTA": 84, "MTN": 85, "NAM": 86,
    "AZE": 87, "MAD": 88, "CGO": 89, "LBA": 90, "LBN": 91,
    "GAB": 92, "TGO": 93, "GAM": 94, "KEN": 95, "CRC": 96,
    "HAI": 97, "ANG": 98, "CUW": 99, "VIN": 100, "CPV": 101,
}


# Last-5 form per code, oldest -> newest. W=win, D=draw, L=loss.
FORM_LAST5: dict[str, list[str]] = {
    "ARG": ["W", "W", "D", "W", "W"],
    "BRA": ["W", "L", "W", "W", "D"],
    "FRA": ["W", "W", "W", "D", "W"],
    "ESP": ["W", "W", "W", "W", "D"],
    "ENG": ["W", "D", "W", "W", "W"],
    "POR": ["W", "W", "D", "W", "W"],
    "GER": ["W", "D", "L", "W", "W"],
    "NED": ["W", "W", "L", "D", "W"],
    "BEL": ["D", "W", "W", "L", "W"],
    "CRO": ["W", "D", "W", "D", "W"],
    "ITA": ["W", "W", "L", "D", "W"],
    "URU": ["L", "W", "D", "W", "L"],
    "COL": ["W", "D", "W", "L", "D"],
    "MAR": ["W", "W", "D", "W", "W"],
    "SUI": ["D", "W", "L", "W", "D"],
    "JPN": ["W", "W", "D", "W", "L"],
    "USA": ["W", "L", "D", "W", "W"],
    "SEN": ["W", "D", "W", "W", "L"],
    "MEX": ["W", "L", "D", "W", "L"],
    "IRN": ["W", "W", "D", "L", "W"],
    "KOR": ["W", "D", "W", "L", "W"],
    "AUT": ["D", "W", "L", "W", "D"],
    "EGY": ["W", "W", "D", "L", "W"],
    "SWE": ["L", "D", "W", "L", "W"],
    "ECU": ["D", "W", "L", "D", "W"],
    "AUS": ["W", "L", "D", "W", "L"],
    "TUR": ["W", "D", "W", "L", "W"],
    "ALG": ["L", "W", "D", "W", "D"],
    "SCO": ["L", "D", "W", "D", "L"],
    "PAR": ["D", "L", "W", "D", "L"],
    "TUN": ["W", "D", "L", "W", "D"],
    "CIV": ["W", "L", "D", "L", "W"],
    "NOR": ["W", "W", "D", "W", "L"],
    "CZE": ["D", "L", "W", "L", "D"],
    "QAT": ["L", "D", "L", "W", "D"],
    "RSA": ["L", "D", "L", "D", "W"],
    "PAN": ["D", "L", "L", "D", "W"],
    "GHA": ["L", "D", "L", "W", "D"],
    "IRQ": ["W", "L", "D", "L", "W"],
    "BIH": ["D", "L", "L", "D", "W"],
    "JOR": ["L", "D", "W", "L", "D"],
    "UZB": ["W", "D", "L", "D", "W"],
    "CAN": ["L", "D", "L", "W", "D"],
    "KSA": ["L", "D", "L", "W", "L"],
    "NZL": ["D", "L", "D", "L", "W"],
    "COD": ["L", "D", "L", "D", "L"],
    "HAI": ["L", "L", "D", "L", "L"],
    "CUW": ["L", "L", "L", "D", "L"],
    "CPV": ["L", "D", "L", "L", "D"],
}


# Venue altitude (metres above sea level) — Dixon-Coles xG adjustment kicks in
# above ~1500m. Only the affected WC 2026 venues are listed.
VENUE_ALTITUDE_M: dict[str, float] = {
    "Estadio Akron, Guadalajara": 1566,
    "Estadio Azteca, Mexico City": 2240,
    "BBVA Stadium, Monterrey": 540,
    # All US/Canada venues are below 1500m and use the default of 0.
}


# Whether the home side in the fixture is actually playing in a host stadium.
# For WC 2026 only Mexico/USA/Canada get true home-venue boosts; other
# nominal "home" sides are neutral.
HOST_NATIONS: set[str] = {"USA", "MEX", "CAN"}


def get_world_ranking(code: str) -> int:
    return WORLD_RANKING.get(code, 50)


def get_form(code: str) -> list[str]:
    return list(FORM_LAST5.get(code, ["D", "D", "D", "D", "D"]))


def get_venue_altitude(venue: str | None) -> float:
    if not venue:
        return 0.0
    return float(VENUE_ALTITUDE_M.get(venue, 0.0))


def get_home_advantage(home_code: str, venue: str | None) -> float:
    """Return [0, 0.18]. Only host nations playing in their own country get
    a meaningful home boost; everyone else gets 0."""
    if home_code not in HOST_NATIONS:
        return 0.0
    if not venue:
        return 0.0
    venue_l = venue.lower()
    if home_code == "USA" and any(c in venue_l for c in ("dallas", "atlanta", "boston", "kansas", "los angeles", "miami", "new york", "philadelphia", "san francisco", "seattle", "houston")):
        return 0.12
    if home_code == "MEX" and any(c in venue_l for c in ("mexico city", "guadalajara", "monterrey")):
        return 0.16
    if home_code == "CAN" and any(c in venue_l for c in ("toronto", "vancouver")):
        return 0.12
    return 0.0
