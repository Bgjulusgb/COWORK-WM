"""Rebuild wm2026_data.json from the worldcup26.ir API snapshot.

Usage:
    cd backend && python scripts/rebuild_wm_data.py

Reads scripts/wc2026_api_data.json (output of sync_from_api.py)
and writes a fresh scripts/wm2026_data.json in the project's format.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

API_DATA = ROOT / "wc2026_api_data.json"
OUT = ROOT / "wm2026_data.json"

# ISO 3166-1 alpha-3 codes for all 48 API teams
NAME_TO_CODE = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR",
    "Czech Republic": "CZE", "Canada": "CAN",
    "Bosnia and Herzegovina": "BIH", "Qatar": "QAT", "Switzerland": "SUI",
    "Brazil": "BRA", "Morocco": "MAR", "Haiti": "HAI", "Scotland": "SCO",
    "United States": "USA", "Australia": "AUS", "Paraguay": "PAR",
    "Turkey": "TUR", "Germany": "GER", "Ivory Coast": "CIV",
    "Ecuador": "ECU", "Curaçao": "CUW", "Curacao": "CUW",
    "Netherlands": "NED", "Japan": "JPN", "Sweden": "SWE", "Tunisia": "TUN",
    "Belgium": "BEL", "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL",
    "Spain": "ESP", "Cape Verde": "CPV", "Saudi Arabia": "KSA",
    "Uruguay": "URU", "France": "FRA", "Senegal": "SEN", "Iraq": "IRQ",
    "Norway": "NOR", "Argentina": "ARG", "Algeria": "ALG",
    "Austria": "AUT", "Jordan": "JOR",
    "Portugal": "POR", "Uzbekistan": "UZB",
    "Democratic Republic of the Congo": "COD", "Colombia": "COL",
    "England": "ENG", "Croatia": "CRO", "Ghana": "GHA", "Panama": "PAN",
}

# Flag emojis
CODE_TO_FLAG = {
    "MEX": "\U0001f1f2\U0001f1fd", "RSA": "\U0001f1ff\U0001f1e6",
    "KOR": "\U0001f1f0\U0001f1f7", "CZE": "\U0001f1e8\U0001f1ff",
    "CAN": "\U0001f1e8\U0001f1e6", "BIH": "\U0001f1e7\U0001f1e6",
    "QAT": "\U0001f1f6\U0001f1e6", "SUI": "\U0001f1e8\U0001f1ed",
    "BRA": "\U0001f1e7\U0001f1f7", "MAR": "\U0001f1f2\U0001f1e6",
    "HAI": "\U0001f1ed\U0001f1f9", "SCO": "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "USA": "\U0001f1fa\U0001f1f8", "AUS": "\U0001f1e6\U0001f1fa",
    "PAR": "\U0001f1f5\U0001f1fe", "TUR": "\U0001f1f9\U0001f1f7",
    "GER": "\U0001f1e9\U0001f1ea", "CIV": "\U0001f1e8\U0001f1ee",
    "ECU": "\U0001f1ea\U0001f1e8", "CUW": "\U0001f1e8\U0001f1fc",
    "NED": "\U0001f1f3\U0001f1f1", "JPN": "\U0001f1ef\U0001f1f5",
    "SWE": "\U0001f1f8\U0001f1ea", "TUN": "\U0001f1f9\U0001f1f3",
    "BEL": "\U0001f1e7\U0001f1ea", "EGY": "\U0001f1ea\U0001f1ec",
    "IRN": "\U0001f1ee\U0001f1f7", "NZL": "\U0001f1f3\U0001f1ff",
    "ESP": "\U0001f1ea\U0001f1f8", "CPV": "\U0001f1e8\U0001f1fb",
    "KSA": "\U0001f1f8\U0001f1e6", "URU": "\U0001f1fa\U0001f1fe",
    "FRA": "\U0001f1eb\U0001f1f7", "SEN": "\U0001f1f8\U0001f1f3",
    "IRQ": "\U0001f1ee\U0001f1f6", "NOR": "\U0001f1f3\U0001f1f4",
    "ARG": "\U0001f1e6\U0001f1f7", "ALG": "\U0001f1e9\U0001f1ff",
    "AUT": "\U0001f1e6\U0001f1f9", "JOR": "\U0001f1ef\U0001f1f4",
    "POR": "\U0001f1f5\U0001f1f9", "UZB": "\U0001f1fa\U0001f1ff",
    "COD": "\U0001f1e8\U0001f1e9", "COL": "\U0001f1e8\U0001f1f4",
    "ENG": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "CRO": "\U0001f1ed\U0001f1f7", "GHA": "\U0001f1ec\U0001f1ed",
    "PAN": "\U0001f1f5\U0001f1e6",
}

# Default stats (will be overridden if we had real data)
DEFAULT_ELO = {
    "ARG": 2130, "FRA": 2120, "BRA": 2080, "ESP": 2060, "ENG": 2050,
    "POR": 2040, "GER": 2020, "NED": 2010, "ITA": 1990, "BEL": 1970,
    "SUI": 1940, "DEN": 1940, "MAR": 1920, "JPN": 1920, "URU": 1920,
    "COL": 1880, "KOR": 1850, "USA": 1850, "AUT": 1860, "SEN": 1860,
    "POL": 1840, "MEX": 1830, "CRO": 1870, "TUR": 1820, "IRN": 1820,
    "SRB": 1820, "UKR": 1810, "ECU": 1800, "AUS": 1760, "SCO": 1760,
    "CIV": 1760, "ALG": 1710, "NOR": 1800, "SWE": 1810, "CZE": 1780,
    "QAT": 1720, "PAR": 1730, "GHA": 1700, "UZB": 1700, "PAN": 1700,
    "CAN": 1750, "TUN": 1720, "EGY": 1780, "RSA": 1640, "NZL": 1620,
    "IRQ": 1700, "JOR": 1660, "HAI": 1570, "BIH": 1760, "KSA": 1730,
    "CPV": 1580, "COD": 1660, "CUW": 1480,
}

DEFAULT_XG = {
    "ARG": (2.05, 0.80), "FRA": (2.00, 0.88), "BRA": (2.10, 0.85),
    "ESP": (1.95, 0.90), "ENG": (1.82, 0.98), "POR": (1.88, 0.95),
    "GER": (1.90, 1.05), "NED": (1.78, 1.02), "BEL": (1.65, 1.10),
    "SUI": (1.62, 1.00), "MAR": (1.45, 0.90), "JPN": (1.60, 1.00),
    "URU": (1.50, 1.08), "COL": (1.52, 1.15), "KOR": (1.45, 1.15),
    "USA": (1.55, 1.20), "AUT": (1.48, 1.15), "SEN": (1.42, 1.05),
    "MEX": (1.48, 1.25), "CRO": (1.41, 1.12), "TUR": (1.40, 1.25),
    "IRN": (1.38, 1.18), "ECU": (1.38, 1.30), "AUS": (1.28, 1.35),
    "SCO": (1.25, 1.35), "CIV": (1.35, 1.25), "ALG": (1.15, 1.32),
    "NOR": (1.55, 1.10), "SWE": (1.45, 1.15), "CZE": (1.38, 1.18),
    "QAT": (1.25, 1.40), "PAR": (1.28, 1.35), "GHA": (1.20, 1.45),
    "UZB": (1.15, 1.38), "PAN": (1.08, 1.48), "CAN": (1.35, 1.30),
    "TUN": (1.18, 1.30), "EGY": (1.32, 1.20), "RSA": (1.05, 1.60),
    "NZL": (0.92, 1.75), "IRQ": (1.20, 1.40), "JOR": (1.10, 1.45),
    "HAI": (0.85, 1.80), "BIH": (1.35, 1.22), "KSA": (1.22, 1.38),
    "CPV": (0.90, 1.70), "COD": (1.15, 1.45), "CUW": (0.75, 1.90),
}

CODE_TO_SUBREDDITS = {
    "MEX": ["LigaMX", "mexico"], "RSA": ["southafrica", "soccer"],
    "KOR": ["soccer", "korea"], "CZE": ["czech", "soccer"],
    "CAN": ["CanadianPL", "canada"], "BIH": ["bih", "soccer"],
    "QAT": ["qatar", "soccer"], "SUI": ["switzerland", "soccer"],
    "BRA": ["futebol", "brasil"], "MAR": ["morocco", "soccer"],
    "HAI": ["haiti", "soccer"], "SCO": ["scotland", "soccer"],
    "USA": ["ussoccer", "MLS"], "AUS": ["aleague", "australia"],
    "PAR": ["paraguay", "soccer"], "TUR": ["turkey", "soccer"],
    "GER": ["bundesliga", "germany"], "CIV": ["cotedivoire", "soccer"],
    "ECU": ["ecuador", "soccer"], "CUW": ["curacao", "soccer"],
    "NED": ["soccer", "thenetherlands"], "JPN": ["japansoccer", "japan"],
    "SWE": ["allsvenskan", "sweden"], "TUN": ["tunisia", "soccer"],
    "BEL": ["belgium", "soccer"], "EGY": ["egypt", "soccer"],
    "IRN": ["iran", "soccer"], "NZL": ["newzealand", "soccer"],
    "ESP": ["laliga", "spain"], "CPV": ["capeverde", "soccer"],
    "KSA": ["saudiarabia", "soccer"], "URU": ["uruguay", "soccer"],
    "FRA": ["lequipe", "soccer"], "SEN": ["senegal", "soccer"],
    "IRQ": ["iraq", "soccer"], "NOR": ["eliteserien", "norway"],
    "ARG": ["argentina", "soccer"], "ALG": ["algeria", "soccer"],
    "AUT": ["austria", "soccer"], "JOR": ["jordan", "soccer"],
    "POR": ["soccer", "portugal"], "UZB": ["uzbekistan", "soccer"],
    "COD": ["congo", "soccer"], "COL": ["colombia", "soccer"],
    "ENG": ["england", "PremierLeague"], "CRO": ["croatia", "soccer"],
    "GHA": ["ghana", "soccer"], "PAN": ["panama", "soccer"],
}

CODE_TO_KEYWORDS = {
    "MEX": ["Mexico", "MEX", "El Tri"],
    "RSA": ["South Africa", "RSA", "Bafana Bafana"],
    "KOR": ["South Korea", "Korea", "KOR", "Taeguk Warriors"],
    "CZE": ["Czech Republic", "Czechia", "CZE"],
    "CAN": ["Canada", "CAN", "Les Rouges"],
    "BIH": ["Bosnia", "Herzegovina", "BIH"],
    "QAT": ["Qatar", "QAT", "Al-Annabi"],
    "SUI": ["Switzerland", "SUI", "Nati"],
    "BRA": ["Brazil", "Brasil", "BRA", "Selecao"],
    "MAR": ["Morocco", "MAR", "Atlas Lions"],
    "HAI": ["Haiti", "HAI"],
    "SCO": ["Scotland", "SCO", "Tartan Army"],
    "USA": ["USA", "USMNT", "United States"],
    "AUS": ["Australia", "AUS", "Socceroos"],
    "PAR": ["Paraguay", "PAR", "Guaranies"],
    "TUR": ["Turkey", "Turkiye", "TUR"],
    "GER": ["Germany", "Deutschland", "GER", "Die Mannschaft"],
    "CIV": ["Ivory Coast", "CIV", "Elephants"],
    "ECU": ["Ecuador", "ECU", "La Tri"],
    "CUW": ["Curacao", "CUW"],
    "NED": ["Netherlands", "Holland", "NED", "Oranje"],
    "JPN": ["Japan", "JPN", "Samurai Blue"],
    "SWE": ["Sweden", "SWE", "Blagult"],
    "TUN": ["Tunisia", "TUN", "Eagles of Carthage"],
    "BEL": ["Belgium", "BEL", "Red Devils"],
    "EGY": ["Egypt", "EGY", "Pharaohs"],
    "IRN": ["Iran", "IRN", "Team Melli"],
    "NZL": ["New Zealand", "NZL", "All Whites"],
    "ESP": ["Spain", "ESP", "La Roja"],
    "CPV": ["Cape Verde", "CPV", "Blue Sharks"],
    "KSA": ["Saudi Arabia", "KSA", "Green Falcons"],
    "URU": ["Uruguay", "URU", "La Celeste"],
    "FRA": ["France", "FRA", "Les Bleus"],
    "SEN": ["Senegal", "SEN", "Lions of Teranga"],
    "IRQ": ["Iraq", "IRQ", "Lions of Mesopotamia"],
    "NOR": ["Norway", "NOR", "Landslaget"],
    "ARG": ["Argentina", "ARG", "Albiceleste"],
    "ALG": ["Algeria", "ALG", "Fennec Foxes"],
    "AUT": ["Austria", "AUT"],
    "JOR": ["Jordan", "JOR"],
    "POR": ["Portugal", "POR", "Selecao"],
    "UZB": ["Uzbekistan", "UZB", "White Wolves"],
    "COD": ["DR Congo", "Congo", "COD", "Leopards"],
    "COL": ["Colombia", "COL", "Los Cafeteros"],
    "ENG": ["England", "ENG", "Three Lions"],
    "CRO": ["Croatia", "CRO", "Vatreni"],
    "GHA": ["Ghana", "GHA", "Black Stars"],
    "PAN": ["Panama", "PAN", "Canaleros"],
}


def _clean_name(name):
    # Handle Curaçao encoding issue
    name = name.replace("Curaçao", "Curaçao").replace("Cura�ao", "Curaçao")
    if "Cura" in name and "ao" in name:
        name = "Curaçao"
    return name.strip()


def main():
    raw = json.loads(API_DATA.read_text(encoding="utf-8"))

    api_teams = raw.get("teams", [])
    api_games = raw.get("games", [])
    api_stadiums = raw.get("stadiums", [])

    # Build team_id -> team_name mapping
    id_to_name = {}
    for t in api_teams:
        name = _clean_name(t.get("name_en", "Unknown"))
        id_to_name[str(t.get("id", ""))] = name

    # Build stadium_id -> info mapping
    stadium_map = {}
    for s in api_stadiums:
        sid = str(s.get("id", ""))
        sname = s.get("name_en", s.get("name", "TBD"))
        city = s.get("city_en", s.get("city", ""))
        stadium_map[sid] = f"{sname}, {city}" if city else sname

    # Build teams dict
    teams_dict = {}
    for t in api_teams:
        name = _clean_name(t.get("name_en", "Unknown"))
        code = NAME_TO_CODE.get(name)
        if not code:
            print(f"[WARN] No code mapping for team: {name}")
            code = name[:3].upper()

        xg = DEFAULT_XG.get(code, (1.20, 1.40))
        teams_dict[code] = {
            "name": name,
            "code": code,
            "fifa_code": code,
            "flag_emoji": CODE_TO_FLAG.get(code, "\U0001f3f3️"),
            "elo_rating": DEFAULT_ELO.get(code, 1650),
            "world_ranking": 50,
            "avg_xg_season": xg[0],
            "avg_xg_conceded": xg[1],
            "form_last5": ["D", "D", "D", "D", "D"],
            "subreddits": CODE_TO_SUBREDDITS.get(code, ["soccer"]),
            "keywords": CODE_TO_KEYWORDS.get(code, [name, code]),
        }

    # Build groups dict
    groups_by_name = {}
    for g in raw.get("groups", []):
        gname = g.get("name", "?")
        team_ids = [str(entry.get("team_id", "")) for entry in g.get("teams", [])]
        team_codes = []
        for tid in team_ids:
            tname = id_to_name.get(tid, "Unknown")
            code = NAME_TO_CODE.get(tname)
            if code:
                team_codes.append(code)
            else:
                print(f"[WARN] Group {gname}: no code for team_id={tid} name={tname}")
        groups_by_name[gname] = team_codes

    # Build schedule
    schedule = {}
    for game in api_games:
        grp = game.get("group", "?")
        if grp not in schedule:
            schedule[grp] = []

        home_name = _clean_name(game.get("home_team_name_en", "?"))
        away_name = _clean_name(game.get("away_team_name_en", "?"))
        home_code = NAME_TO_CODE.get(home_name, home_name[:3].upper())
        away_code = NAME_TO_CODE.get(away_name, away_name[:3].upper())

        # Parse date "06/15/2026 12:00" -> ISO
        local_date = game.get("local_date", "")
        try:
            from datetime import datetime
            dt = datetime.strptime(local_date, "%m/%d/%Y %H:%M")
            kickoff = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            kickoff = "2026-06-11T00:00:00Z"

        stadium_id = str(game.get("stadium_id", ""))
        venue = stadium_map.get(stadium_id, "TBD")

        match_number = int(game.get("id", 0))

        schedule[grp].append({
            "home": home_code,
            "away": away_code,
            "match_number": match_number,
            "kickoff_utc": kickoff,
            "venue": venue,
        })

    # Sort matches within each group by match_number
    for grp in schedule:
        schedule[grp].sort(key=lambda m: m["match_number"])

    result = {
        "teams": teams_dict,
        "groups": groups_by_name,
        "group_schedule": schedule,
    }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"  Teams: {len(teams_dict)}")
    print(f"  Groups: {len(groups_by_name)}")
    print(f"  Games: {sum(len(v) for v in schedule.values())}")


if __name__ == "__main__":
    main()
