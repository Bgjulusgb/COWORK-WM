"""Validity / transparency endpoint.

Surfaces the model architecture, data sources, and confidence caveats so
clients (dashboard, audits, downstream consumers) can self-document what
they're showing.
"""
from __future__ import annotations

from fastapi import APIRouter

from config.settings import settings

router = APIRouter(prefix="/api/model", tags=["model"])


# Tight emotion-label definitions so "kontrovers", "euphorisch" etc. have a
# documented derivation rather than being magic strings. Aligned with the
# logic in analysis/advanced_metrics.py.
EMOTION_DEFINITIONS = {
    "euphorisch": "Mittelwert(ensemble) ≥ +0.5 mit hype_ratio ≥ 0.4 — viele klar positive Beiträge.",
    "optimistisch": "Mittelwert(ensemble) zwischen +0.2 und +0.5 mit positivem Trend.",
    "neutral": "Mittelwert(ensemble) in [−0.2, +0.2] und niedrige Streuung.",
    "kontrovers": "|polarization| > 0.6 oder Fanbalance < 0.4 — Lager spalten sich stark.",
    "skeptisch": "Mittelwert(ensemble) zwischen −0.5 und −0.2; oft mit hohem cope_ratio.",
    "frustriert": "Mittelwert(ensemble) ≤ −0.5 mit dominanten Emotionen anger/disgust.",
    "ängstlich": "Dominante Emotion fear in Tier-1/2-Posts mit hohem engagement.",
    "ruhig": "Geringe Post-Velocity (< 2/h) und enge Standardabweichung.",
}


SAMPLE_SIZE_GUIDANCE = {
    "reliable_min_posts": 500,
    "low_warning_threshold": 100,
    "rationale": (
        "Bei < 500 Posts liegen Konfidenz-Intervalle für die Mittelwert-Sentiment-Schätzung "
        "üblicherweise über ±0.05, was die 1X2-Wahrscheinlichkeit deutlich verschiebt. "
        "Unter 100 Posts ist der Mittelwert dominiert durch wenige Power-User und nicht "
        "repräsentativ für die Fanbase."
    ),
}


# Per-factor target weights + a short rationale. Weights read live from
# settings so this stays in sync with the actual ensemble configuration.
_FACTOR_INFO: list[tuple[str, str, str]] = [
    ("elo_strength", "factor_weight_elo", "Elo-Delta → λ-Tilt (clip ±400, ±20 % xG bei 200 Elo)."),
    ("form", "factor_weight_form", "Rezenz-/Tier-gewichtete Punkte-Rate der letzten Spiele (Hvattum 2010)."),
    ("head_to_head", "factor_weight_h2h", "Bilanz der letzten Begegnungen, Bayes-Shrinkage bei kleinem N."),
    ("goal_efficiency", "factor_weight_goals", "xG-Proxy aus Tor-/Gegentorraten (Maher 1982), geom. normiert."),
    ("tournament_context", "factor_weight_context", "Host-Nation-Heimvorteil (USA/MEX/CAN) + KO-Dämpfung."),
    ("sentiment", "factor_weight_sentiment", "Reddit-VADER/TextBlob/(RoBERTa) + Momentum, Sample-gedämpft."),
    ("squad_availability", "factor_weight_squad", "Kader/Absenzen (Wikidata, best-effort)."),
    ("fifa_ranking", "factor_weight_fifa_rank", "FIFA-Ranking-Delta, komplementär zu Elo."),
    ("rest_travel", "factor_weight_rest_travel", "Ruhetage-Delta + Reise/Jetlag (Haversine + Zeitzonen)."),
    ("venue_altitude", "factor_weight_altitude", "Globaler Tor-Modifier: Ausdauer-Malus > 1500 m (McSharry 2007)."),
    ("market_odds", "factor_weight_market", "Vig-korrigierte Quoten → λ-Tilt (Markteffizienz)."),
    ("weather", "factor_weight_weather", "Globaler Tor-Modifier: Hitze/Feuchte dämpft Tore (Link 2017)."),
    ("injury_news", "factor_weight_injury", "RSS-Verletzungs-News (BBC/Guardian/ESPN), Keyword-Severity."),
    ("momentum_drift", "factor_weight_momentum", "Sentiment-Trend-Slope (Änderungsrate, nicht Niveau)."),
    ("ml_blend", "factor_weight_ml", "Optionaler trainierter xG-Regressor (aus, bis Artefakt trainiert)."),
    # v3.3 — neue Faktoren.
    ("ml_blend_lgbm", "factor_weight_ml_lgbm", "LightGBM-Variante des ML-Blends (zweiter ML-Head)."),
    ("llm_sentiment", "factor_weight_llm_sentiment", "NVIDIA build.nvidia.com LLM Aspect-Sentiment (attack/defence/morale)."),
    ("lineup_strength", "factor_weight_lineup", "Aufstellungsstärke (FotMob/SofaScore) vs. Saison-Schnitt-XI (Peeters 2018)."),
    ("squad_value", "factor_weight_squad_value", "Log-Verhältnis Transfermarkt-Marktwert Top-11 (wisdom-of-crowds)."),
    ("network_strength", "factor_weight_network", "PageRank über Match-Graph (Bryan & Leise 2006)."),
]


@router.get("")
async def model_info():
    factors = [
        {
            "name": name,
            "target_weight": getattr(settings, attr),
            "active": getattr(settings, attr) > 0,
            "kind": "global" if name in ("weather", "venue_altitude") else "tilt",
            "method": method,
        }
        for name, attr, method in _FACTOR_INFO
    ]
    return {
        "version": "3.3",
        "summary": (
            "Multi-Faktor-Ensemble: bis zu 18 unabhängige Signale (v3.3) liefern home/away-"
            "Stärken, die (re-normalisiert) zu λ-Multiplikatoren auf das Basis-xG verrechnet "
            "werden. Tor-Modell wählbar: Dixon-Coles-Poisson, Negative-Binomial oder "
            "statsmodels-GLM-Poisson. Globale Modifier (Wetter/Höhe) dämpfen beide λ nach der "
            "Mittelung; Bookmaker-Quoten gehen als λ-Tilt ODER als 1X2-Prior ein (nicht doppelt). "
            "Neu in v3.3: NVIDIA-LLM-Aspect-Sentiment, FBref/Understat-xG, FotMob/SofaScore-"
            "Lineups, Transfermarkt-Marktwerte, PageRank-Netzwerkstärke, Optuna-Gewichts-Tuning."
        ),
        "components": {
            "factor_ensemble": {
                "factors": factors,
                "active_count": sum(1 for f in factors if f["active"]),
                "renormalisation": "Fehlt eine Datenquelle, wird der Faktor übersprungen und die übrigen Gewichte auf Summe 1 skaliert.",
                "global_modifiers": "weather + venue_altitude wirken symmetrisch (Tor-Total), multiplikativ nach der Mittelung, gedeckelt auf −18 %.",
                "confidence": "0.6 · Ø(Faktor-Konfidenz) + 0.4 · Übereinstimmung(1 − stdev der Tilt-Ratios).",
            },
            "goal_model": {
                "type": settings.goal_model,
                "negbin_size": settings.negbin_size,
                "dixon_coles": "ρ-Korrektur für 0-0/1-0/0-1/1-1; Negative-Binomial optional für Tor-Überdispersion.",
            },
            "sentiment_ensemble": {
                "models": ["VADER", "TextBlob", "RoBERTa (optional)"],
                "base_weights": {"vader": 0.55, "textblob": 0.25, "roberta": 0.20},
                "feeds": "SentimentFactor + MomentumDriftFactor",
                "language_handling": "Englisch-trainierte Modelle bei nicht-englischen Texten heruntergewichtet.",
            },
            "data_sources": {
                "note": "Live-first, jede Quelle fällt bei Fehlern auf einen deterministischen Mock zurück. Provenance (live/cache/mock) hängt an jedem FactorSignal.",
                "connectors": {
                    "openfootball": {"data": "WM-History 2018/2022 + 2026-Fixtures", "key": False, "mock": settings.use_mock_openfootball},
                    "thesportsdb": {"data": "Team-Meta/Venue", "key": "public test key", "mock": settings.use_mock_thesportsdb},
                    "openligadb": {"data": "Ergänzende History", "key": False, "mock": settings.use_mock_openligadb},
                    "wikidata": {"data": "Kader/Absenzen", "key": False, "mock": settings.use_mock_wikidata},
                    "open-meteo": {"data": "Wetter zur Anstoßzeit", "key": False, "mock": settings.use_mock_weather},
                    "rss_news": {"data": "BBC/Guardian/ESPN Verletzungs-News", "key": False, "mock": settings.use_mock_rss},
                    "odds": {"data": "the-odds-api.com", "key": "ODDS_API_KEY", "enabled": settings.enable_odds_integration},
                },
                "reddit_tiers": [
                    {"tier": 1, "examples": ["worldcup", "soccer"], "weight": 1.0},
                    {"tier": 2, "examples": ["country-specific subs"], "weight": 1.5},
                    {"tier": 3, "examples": ["national subs"], "weight": 0.8},
                ],
                "live_scores": "worldcup26.ir Open Data API",
            },
            "kickoff_timezone": "Alle kickoff_utc-Werte sind UTC. Das Frontend rendert in der lokalen Browser-Zeitzone.",
        },
        "feature_flags": {
            "use_factor_ensemble": settings.use_factor_ensemble,
            "goal_model": settings.goal_model,
            "use_roberta": settings.use_roberta,
            "enable_odds_integration": settings.enable_odds_integration,
            "enable_h2h": settings.enable_h2h,
            "enable_twitter_crawler": settings.enable_twitter_crawler,
            "enable_scheduler": settings.enable_scheduler,
        },
        "emotion_definitions": EMOTION_DEFINITIONS,
        "sample_size_guidance": SAMPLE_SIZE_GUIDANCE,
        "validity_caveats": [
            "Faktoren ohne Live-Daten laufen auf deterministischen Mocks — die Quellen-Badge (live/cache/mock) zeigt es pro Signal.",
            "Sentiment-Stichproben < 500 Posts liefern instabile Mittelwerte; die Konfidenz wird entsprechend gedämpft.",
            "Elo/Form/FIFA-Ranking sind Snapshots (team_real_data, Stand 2026-05-31); via scripts/refresh_elo.py bzw. update_yaml_team_data.py aktualisierbar.",
            "Predictions sind explorativ und nicht als Wett-Beratung gedacht.",
        ],
    }
