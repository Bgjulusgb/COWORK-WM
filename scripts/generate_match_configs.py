"""Generate YAML match configs for all WM 2026 group-stage matches.

Usage (from backend/ directory):
    python scripts/generate_match_configs.py

Creates one YAML per match in backend/config/matches/group_X/.
Skips already-existing files to avoid overwriting manual edits.
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "wm2026_data.json"
OUT_DIR = ROOT / "config" / "matches"

# Kickoff times per match in group stage (index in group: 0..5)
# matches within a group alternate venues
SEARCH_KEYWORDS_GLOBAL = ["world cup 2026", "worldcup", "WC2026", "FIFA2026"]


def load_data() -> dict:
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def team_keywords(team: dict) -> list[str]:
    return team.get("keywords", [team["name"], team["code"]])


def make_yaml(group: str, match_no: int, home: dict, away: dict,
              kickoff: str, venue: str) -> dict:
    home_code = home["code"].lower()
    away_code = away["code"].lower()
    match_id = f"wm2026_group{group.lower()}_{home_code}_vs_{away_code}"

    home_subs = home.get("subreddits", ["soccer"])
    away_subs = away.get("subreddits", ["soccer"])

    tier1 = [
        {"subreddit": "worldcup", "language_filter": "en", "min_post_score": 10,
         "search_keywords": [home["name"].lower(), away["name"].lower(),
                             home["code"].lower(), away["code"].lower(),
                             f"{home['code'].lower()} vs {away['code'].lower()}"]},
        {"subreddit": "soccer", "language_filter": "en", "min_post_score": 5,
         "search_keywords": [home["name"].lower(), away["name"].lower(),
                             "world cup", "prediction", "preview"]},
    ]

    tier2_home = [{"subreddit": s, "min_post_score": 3, "include_comments": True,
                   "comment_depth": 2} for s in home_subs]
    tier2_away = [{"subreddit": s, "min_post_score": 3, "include_comments": True,
                   "comment_depth": 2} for s in away_subs]

    # Minimal stopwords except for team names to avoid false attribution
    home_kw = team_keywords(home)
    away_kw = team_keywords(away)

    return {
        "match": {
            "id": match_id,
            "tournament": "FIFA World Cup 2026",
            "group": group,
            "match_number": match_no,
            "phase": "group_stage",
            "kickoff_utc": kickoff,
            "venue": venue,
        },
        "teams": {
            "home": {
                "name": home["name"],
                "code": home["code"],
                "fifa_code": home.get("fifa_code", home["code"]),
                "flag_emoji": home["flag_emoji"],
                "elo_rating": home["elo_rating"],
                "avg_xg_season": home["avg_xg_season"],
                "avg_xg_conceded": home["avg_xg_conceded"],
                "form_last5": home["form_last5"],
                "world_ranking": home.get("world_ranking", 50),
            },
            "away": {
                "name": away["name"],
                "code": away["code"],
                "fifa_code": away.get("fifa_code", away["code"]),
                "flag_emoji": away["flag_emoji"],
                "elo_rating": away["elo_rating"],
                "avg_xg_season": away["avg_xg_season"],
                "avg_xg_conceded": away["avg_xg_conceded"],
                "form_last5": away["form_last5"],
                "world_ranking": away.get("world_ranking", 50),
            },
        },
        "reddit_sources": {
            "tier1_global": tier1,
            "tier2_team_specific": {"home": tier2_home, "away": tier2_away},
            "tier3_national_sentiment": {
                "home": [{"subreddit": home_subs[0], "search_keywords": SEARCH_KEYWORDS_GLOBAL}],
                "away": [{"subreddit": away_subs[0], "search_keywords": SEARCH_KEYWORDS_GLOBAL}],
            },
        },
        "crawl_config": {
            "filters": {
                "min_account_age_days": 30,
                "min_post_score": 3,
                "min_comment_score": 1,
                "exclude_flairs": ["Meme", "Satire", "Off-Topic"],
                "exclude_bots": True,
                "language": "en",
                "max_post_age_hours": 72,
            }
        },
        "preprocessing": {
            "text_cleaning": {
                "remove_urls": True, "remove_reddit_markup": True,
                "remove_mentions": True, "remove_subreddit_refs": True,
                "normalize_unicode": True, "expand_contractions": True,
                "lowercase": True,
            },
            "sport_slang_expansion": {
                "enable": True,
                "custom_dict": {
                    "W": "win", "L": "loss", "brace": "two goals",
                    "hat trick": "three goals", "clean sheet": "no goals conceded",
                    "pen": "penalty", "og": "own goal", "nil": "zero",
                    "bottled": "lost under pressure", "class": "excellent",
                    "quality": "excellent", "garbage": "very bad",
                    "trash": "very bad", "banging": "great",
                    "worldie": "spectacular goal", "park the bus": "defensive",
                },
            },
            "tokenization": {
                "library": "spacy", "model": "en_core_web_sm",
                "lemmatize": True, "remove_stopwords": True,
                "preserve_negations": True, "min_token_length": 2,
                "max_token_length": 50,
            },
        },
        "sentiment_config": {
            "ensemble_weights": {"vader": 0.55, "textblob": 0.25, "roberta": 0.20},
            "vader_config": {
                "use_social_media_mode": True,
                "custom_lexicon": {
                    "dominate": 2.5, "demolished": -3.0, "clinical": 2.0,
                    "unlucky": -1.5, "deserved": 1.5, "bottled": -2.5,
                    "worldie": 3.5, "garbage": -3.0,
                },
            },
            "team_attribution": {
                "method": "keyword",
                "home_keywords": home_kw,
                "away_keywords": away_kw,
                "unattributed_handling": "neutral",
            },
            "aggregation": {
                "time_windows": [6, 12, 24, 48, 72],
                "engagement_weighting": {
                    "formula": "log(1+score)*log(1+num_comments)*upvote_ratio",
                    "normalize": True,
                },
                "momentum_calculation": {"window_hours": 6},
            },
        },
        "prediction_config": {
            "goal_prediction": {
                "method": "heuristic_xg_dixoncoles",
                "dixon_coles_rho": 0.1,
                "max_goals_display": 6,
                "markets_to_calculate": ["1X2", "over_under_25", "over_under_15",
                                         "over_under_35", "btts", "exact_score"],
            },
            "best_bet": {
                "min_probability": 0.55,
                "min_confidence": 0.60,
                "max_bets_per_match": 3,
            },
        },
    }


def _placeholder_team(code: str, label: str) -> dict:
    """Synthetic team entry for KO-round placeholders (TBD slots)."""
    return {
        "name": label,
        "code": code,
        "fifa_code": code,
        "flag_emoji": "🏳️",
        "elo_rating": 1750,
        "avg_xg_season": 1.40,
        "avg_xg_conceded": 1.40,
        "form_last5": ["D", "D", "D", "D", "D"],
        "world_ranking": 50,
        "subreddits": ["worldcup", "soccer"],
        "keywords": [label, code],
    }


def _ko_template(phase_name: str, dir_name: str, slots: list[tuple[str, str, str, str]],
                 kickoff_base: str, match_no_start: int) -> list[tuple[str, str, dict, dict, str, int]]:
    """Produce (group_label, filename_prefix, home_dict, away_dict, kickoff_iso, match_no) tuples
    for a knockout round. slots = [(code_home, label_home, code_away, label_away), ...]"""
    out = []
    from datetime import datetime, timedelta

    base = datetime.fromisoformat(kickoff_base.replace("Z", "+00:00"))
    for i, (ch, lh, ca, la) in enumerate(slots):
        kickoff = (base + timedelta(hours=3 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append((
            phase_name,
            f"{dir_name}_{i + 1}",
            _placeholder_team(ch, lh),
            _placeholder_team(ca, la),
            kickoff,
            match_no_start + i,
        ))
    return out


def _ko_make_yaml(phase: str, match_no: int, home: dict, away: dict,
                  kickoff: str, venue: str, phase_label: str) -> dict:
    doc = make_yaml(group=phase_label, match_no=match_no, home=home, away=away,
                    kickoff=kickoff, venue=venue)
    doc["match"]["phase"] = phase
    # Knockout matches use a synthetic id namespace
    doc["match"]["id"] = f"wm2026_{phase}_{home['code'].lower()}_vs_{away['code'].lower()}_{match_no}"
    return doc


# Knockout round skeleton — replace placeholder codes via a follow-up script
# once group winners are known. Match numbers continue past 72.
_KO_SLOTS = {
    "round_of_32": (
        "round_of_32",
        # (home_code, home_label, away_code, away_label)
        [
            ("A1", "Sieger Gruppe A", "TPB", "Bester Gruppendritter B/E/F"),
            ("C1", "Sieger Gruppe C", "TPA", "Bester Gruppendritter A/D/H"),
            ("B1", "Sieger Gruppe B", "F2", "Zweiter Gruppe F"),
            ("D1", "Sieger Gruppe D", "TPC", "Bester Gruppendritter C/G/J"),
            ("E1", "Sieger Gruppe E", "TPD", "Bester Gruppendritter D/I/L"),
            ("G1", "Sieger Gruppe G", "H2", "Zweiter Gruppe H"),
            ("F1", "Sieger Gruppe F", "I2", "Zweiter Gruppe I"),
            ("H1", "Sieger Gruppe H", "TPE", "Bester Gruppendritter F/I/J"),
            ("I1", "Sieger Gruppe I", "L2", "Zweiter Gruppe L"),
            ("J1", "Sieger Gruppe J", "K2", "Zweiter Gruppe K"),
            ("K1", "Sieger Gruppe K", "TPF", "Bester Gruppendritter A/B/G"),
            ("L1", "Sieger Gruppe L", "TPG", "Bester Gruppendritter C/E/H"),
            ("A2", "Zweiter Gruppe A", "B2", "Zweiter Gruppe B"),
            ("C2", "Zweiter Gruppe C", "D2", "Zweiter Gruppe D"),
            ("E2", "Zweiter Gruppe E", "G2", "Zweiter Gruppe G"),
            ("J2", "Zweiter Gruppe J", "TPH", "Bester Gruppendritter B/C/F"),
        ],
        "2026-07-04T18:00:00Z",
        73,
        "ROF32",
    ),
    "round_of_16": (
        "round_of_16",
        [(f"R16H{i+1}", f"Sieger R32 #{2*i+1}", f"R16A{i+1}", f"Sieger R32 #{2*i+2}")
         for i in range(8)],
        "2026-07-11T18:00:00Z",
        89,
        "ROF16",
    ),
    "quarter_finals": (
        "quarter_finals",
        [(f"QF_H{i+1}", f"Sieger Achtelfinale #{2*i+1}", f"QF_A{i+1}", f"Sieger Achtelfinale #{2*i+2}")
         for i in range(4)],
        "2026-07-15T18:00:00Z",
        97,
        "QF",
    ),
    "semi_finals": (
        "semi_finals",
        [("SF_H1", "Sieger Viertelfinale #1", "SF_A1", "Sieger Viertelfinale #2"),
         ("SF_H2", "Sieger Viertelfinale #3", "SF_A2", "Sieger Viertelfinale #4")],
        "2026-07-21T20:00:00Z",
        101,
        "SF",
    ),
    "third_place": (
        "third_place",
        [("TP_H", "Verlierer Halbfinale #1", "TP_A", "Verlierer Halbfinale #2")],
        "2026-07-25T18:00:00Z",
        103,
        "TP",
    ),
    "final": (
        "final",
        [("F_H", "Sieger Halbfinale #1", "F_A", "Sieger Halbfinale #2")],
        "2026-07-26T19:00:00Z",
        104,
        "F",
    ),
}


def _generate_ko_rounds() -> int:
    created = 0
    for phase_key, (dir_name, slots, kickoff_base, match_no_start, label) in _KO_SLOTS.items():
        ko_dir = OUT_DIR / dir_name
        ko_dir.mkdir(parents=True, exist_ok=True)
        rows = _ko_template(phase_key, dir_name, slots, kickoff_base, match_no_start)
        for phase_name, fname, home, away, kickoff, match_no in rows:
            filename = f"{fname}_{home['code']}_vs_{away['code']}.yaml".lower()
            out_path = ko_dir / filename
            if out_path.exists():
                continue
            doc = _ko_make_yaml(
                phase=phase_name,
                match_no=match_no,
                home=home,
                away=away,
                kickoff=kickoff,
                venue="TBD",
                phase_label=label,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            print(f"[OK]   {out_path.relative_to(ROOT)}")
            created += 1
    return created


def main() -> None:
    data = load_data()
    teams = data["teams"]
    schedule = data["group_schedule"]

    created = 0
    skipped = 0
    errors = 0

    for group, matches in schedule.items():
        group_dir = OUT_DIR / f"group_{group.lower()}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for match_info in matches:
            home_code = match_info["home"]
            away_code = match_info["away"]

            if home_code not in teams:
                print(f"[WARN] Team '{home_code}' not found in teams dict — skipping match")
                errors += 1
                continue
            if away_code not in teams:
                print(f"[WARN] Team '{away_code}' not found in teams dict — skipping match")
                errors += 1
                continue

            home = teams[home_code]
            away = teams[away_code]

            filename = f"{home_code.lower()}_vs_{away_code.lower()}.yaml"
            out_path = group_dir / filename

            if out_path.exists():
                print(f"[SKIP] {out_path.relative_to(ROOT)} (already exists)")
                skipped += 1
                continue

            doc = make_yaml(
                group=group,
                match_no=match_info["match_number"],
                home=home,
                away=away,
                kickoff=match_info["kickoff_utc"],
                venue=match_info["venue"],
            )

            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

            print(f"[OK]   {out_path.relative_to(ROOT)}")
            created += 1

    ko_created = _generate_ko_rounds()
    print(f"\nDone: {created} group matches created, {ko_created} KO placeholders created, "
          f"{skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
