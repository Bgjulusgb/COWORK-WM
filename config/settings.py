from pathlib import Path
from typing import List
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def reload_runtime_weights(self, path: Path | None = None) -> dict[str, float]:
        """v3.3 — load tuned/admin-edited factor weights from a YAML artifact.

        Default path: ``models_ml/artifacts/runtime_weights.yaml`` (gitignored).
        When the Admin-Panel PATCHes weights, or when ``scripts/tune_weights``
        writes the Optuna result, the values are merged onto this Settings
        instance. Passing ``path`` lets callers (e.g. the admin endpoint just
        after writing the file) point at the exact artifact they produced —
        helpful for tests using a tmp_path. Missing file = silent no-op.
        """
        path = path or (self.base_dir / "models_ml" / "artifacts" / "runtime_weights.yaml")
        applied: dict[str, float] = {}
        if not path.exists():
            return applied
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return applied
        for key, value in (data or {}).items():
            if not isinstance(key, str) or not key.startswith("factor_weight_"):
                continue
            if not hasattr(self, key):
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            object.__setattr__(self, key, max(0.0, min(1.0, fval)))
            applied[key] = fval
        return applied

    def reload_runtime_flags(self, path: Path | None = None) -> dict[str, object]:
        """v3.6 — load admin-edited boolean/string flags (use_mock_*, goal_model,
        use_nvidia_llm, goal_model_combine) from a separate YAML artifact, so
        a `.env` rollback is one file delete away.

        Default path: ``models_ml/artifacts/runtime_flags.yaml`` (gitignored).
        Only keys that already exist on this Settings instance and match an
        allowed prefix are applied; everything else is silently ignored.
        """
        path = path or (self.base_dir / "models_ml" / "artifacts" / "runtime_flags.yaml")
        applied: dict[str, object] = {}
        if not path.exists():
            return applied
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return applied
        allowed_prefixes = ("use_mock_", "use_", "goal_model", "bootstrap_")
        for key, value in (data or {}).items():
            if not isinstance(key, str):
                continue
            if not hasattr(self, key):
                continue
            if not any(key.startswith(p) or key == p.rstrip("_") for p in allowed_prefixes):
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    new_val: object = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    new_val = int(value)
                elif isinstance(current, float):
                    new_val = float(value)
                else:
                    new_val = str(value)
            except (TypeError, ValueError):
                continue
            object.__setattr__(self, key, new_val)
            applied[key] = new_val
        return applied

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "RedditOrakel/2.0"
    reddit_username: str = ""
    reddit_password: str = ""

    database_url: str = "sqlite+aiosqlite:///./redditorakel.db"

    use_mock_crawler: bool = True
    use_arctic_shift: bool = False
    use_roberta: bool = False
    # IMPROVE-03: swap the emotion model for a Twitter-trained social-media
    # sentiment model. Drop-in compatible; flip via env when ready.
    roberta_model: str = "j-hartmann/emotion-english-distilroberta-base"

    enable_scheduler: bool = False
    scheduler_interval_hours: int = 6
    scheduler_lookahead_hours: int = 36
    scheduler_min_gap_minutes: int = 30

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    log_level: str = "INFO"

    # IMPROVE-15: in-memory cache TTLs.
    cache_match_list_ttl_s: int = 60
    cache_prediction_ttl_s: int = 300

    # IMPROVE-17: API rate limits per IP.
    rate_limit_get_per_minute: int = 60
    rate_limit_post_per_minute: int = 5
    admin_api_key: str = ""

    # EXTEND-02: bookmaker odds integration.
    odds_api_key: str = ""
    enable_odds_integration: bool = False

    # EXTEND-03: H2H data.
    enable_h2h: bool = True

    # EXTEND-01: Twitter/X.
    twitter_bearer_token: str = ""
    enable_twitter_crawler: bool = False

    # MULTIFACTOR-01: master switch for the new factor-ensemble predictor.
    # When false the legacy heuristic xG → Dixon-Coles path runs (BUG-03 + EXTEND-03).
    use_factor_ensemble: bool = True

    # MULTIFACTOR-v3: goal model. "poisson" = Dixon-Coles Poisson (default);
    # "negbin" = negative-binomial marginals for football's goal over-dispersion;
    # "glm_poisson" = v3.3 statsmodels GLM with team/home fixed effects.
    goal_model: str = "poisson"
    negbin_size: float = 8.0    # NB dispersion r — higher ⇒ closer to Poisson

    # v3.6 — Multi-Model-Ensemble Steuerung.
    # "blend"   = gewichtetes Mittel der 3 Modelle als primaere Vorhersage;
    # "primary" = das per goal_model gewaehlte Modell bestimmt die primaere Vorhersage,
    #             die anderen werden trotzdem ausgegeben (Admin-Panel-Vergleich).
    goal_model_combine: str = "blend"
    # Phase-4 (Verbesserungsplan 2.2): BivariatePoisson als 4. Blend-Modell.
    # OFF by default — der Default-Output bleibt unverändert; bei true schaltet
    # resolve_blend_weights() auf BLEND_WEIGHTS_WITH_BIVARIATE um.
    include_bivariate: bool = False
    bivariate_lambda3: float = 0.12   # Kovarianz-Anteil λ₃ (Karlis-Ntzoufras)
    # Bootstrap-CIs um Predictions. n=0 deaktiviert die Berechnung.
    bootstrap_n: int = 500
    bootstrap_xg_sigma: float = 0.15

    # Phase-4 (Verbesserungsplan 2.5): Aggregation der Faktor-Multiplikatoren.
    # "arith" (Default) = gewichtetes arithmetisches Mittel (Bestand);
    # "geom"            = log-linear, exp(Σ wᵢ·ln sᵢ) — konsistent fuer
    #                     multiplikative Tilts und symmetrisch in home/away.
    lambda_aggregation: str = "arith"

    # Phase-5 calibration. --calibrate market anchors the 1X2 toward the vig-free
    # market consensus by this weight (0 = pure model, 1 = pure market). 0.5 is a
    # neutral default; the market is the canonical calibrated forecaster
    # (Constantinou & Fenton 2013).
    calibration_market_weight: float = 0.5

    # Halftime λ share for the HT/FT market (first-half goals are slightly rarer).
    ht_lambda_share: float = 0.45

    # Dixon-Coles low-score correction ρ. 0.1 is the shipped default; tune offline
    # against RPS with scripts/tune_models_offline.py and set the best value here.
    dixon_coles_rho: float = 0.1

    # Dixon-Coles MLE base-xG estimator (analysis/xg_estimator.py). OFF by default
    # so the naive YAML base xG (and thus the default output) is unchanged; turn
    # on to fit attack/defence + home advantage from time-decayed history.
    use_mle_xg: bool = False
    mle_time_decay_xi: float = 0.0065     # exp(-ξ·Δt_days); 0.0065 ≈ ~2-year half-life
    mle_min_matches: int = 6
    mle_max_iter: int = 200

    # MULTIFACTOR-02: external data-source toggles. Defaults assume the live
    # endpoints; flip to true for offline / CI runs.
    use_mock_openfootball: bool = False
    use_mock_thesportsdb: bool = False
    use_mock_openligadb: bool = False
    # Wikidata domain is not in the existing network allowlist, so the squad
    # factor stays mocked by default until the user opts in.
    use_mock_wikidata: bool = True

    # MULTIFACTOR-03: per-factor weights. The ensemble re-normalises whenever
    # a factor reports available=false, so these are *targets*, not hard ratios.
    factor_weight_elo: float = 0.30
    factor_weight_form: float = 0.20
    factor_weight_h2h: float = 0.15
    factor_weight_goals: float = 0.15
    factor_weight_context: float = 0.10
    factor_weight_sentiment: float = 0.10
    # Squad runs on the deterministic Wikidata mock (top-tier sides only); it
    # self-disables and re-normalises out when no squad data is parseable.
    factor_weight_squad: float = 0.05

    # MULTIFACTOR-v3: additional factor weights. All relative targets — the
    # ensemble re-normalises whatever is available, so these need not sum to 1.
    factor_weight_fifa_rank: float = 0.05      # FIFA ranking delta (complements Elo)
    factor_weight_rest_travel: float = 0.06    # rest days + travel / timezone fatigue
    factor_weight_altitude: float = 0.05       # high-altitude stamina malus
    factor_weight_market: float = 0.10         # bookmaker-implied probabilities
    factor_weight_weather: float = 0.04        # heat / humidity goal damping
    factor_weight_injury: float = 0.06         # RSS injury-news impact
    factor_weight_momentum: float = 0.05       # sentiment trend / drift
    # ML blend (EXTEND-05). 0 = off; self-disables anyway when no trained
    # artifact exists (models_ml/artifacts/xg_predictor.json). Set > 0 after
    # running scripts/train_xg_predictor.py to let the model nudge xG.
    factor_weight_ml: float = 0.00
    # v3.3 — second ML head (LightGBM); needs models_ml/artifacts/xg_predictor_lgbm.txt.
    factor_weight_ml_lgbm: float = 0.00

    # v3.3 — additional factors. All start dormant or low so the existing
    # ensemble balance is unchanged until the user activates them via the
    # Admin-Panel or Optuna tuning artifact.
    factor_weight_llm_sentiment: float = 0.00      # NVIDIA-LLM aspect sentiment
    factor_weight_lineup: float = 0.00             # confirmed lineup vs. season-average
    factor_weight_squad_value: float = 0.00        # Transfermarkt market-value ratio
    factor_weight_network: float = 0.00            # PageRank over match graph

    # MULTIFACTOR-04: shared connector knobs. TTLs are upper bounds — connectors
    # may shorten per endpoint (live fixtures: 6h, historical results: 30d).
    datasource_cache_ttl_hours: int = 6
    datasource_http_timeout_s: float = 15.0
    datasource_retry_attempts: int = 3
    datasource_retry_backoff_s: float = 1.5
    # Optional: point at a local openfootball repo clone for fully offline runs.
    openfootball_local_clone: str = ""

    # MULTIFACTOR-v3: new data-source toggles + creds. Live-first per the user's
    # choice; each connector still degrades to a deterministic mock on failure.
    use_mock_clubelo: bool = False
    use_mock_weather: bool = False
    use_mock_rss: bool = False
    # football-data.org needs a free key; mock until one is provided.
    use_mock_football_data: bool = True
    football_data_api_key: str = ""

    # v3.3 — new scraper toggles. Mock-first so CI stays green.
    use_mock_fbref: bool = True
    use_mock_understat: bool = True
    use_mock_fotmob: bool = True
    use_mock_sofascore: bool = True
    use_mock_transfermarkt: bool = True
    # v3.3 — Transfermarkt politeness (HTML scraper).
    transfermarkt_request_gap_s: float = 1.5
    transfermarkt_concurrency: int = 2

    # v3.3 — NVIDIA-LLM (build.nvidia.com, OpenAI-compatible /v1/chat/completions).
    nvidia_api_key: str = ""
    nvidia_llm_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    use_nvidia_llm: bool = False
    llm_max_posts_per_tier: int = 6     # cap LLM token budget per match
    llm_request_budget_per_match: int = 3
    llm_temperature: float = 0.1        # near-deterministic for sentiment scoring

    # v3.3 — orchestration backend. APScheduler is the default; Prefect is opt-in.
    use_prefect: bool = False

    base_dir: Path = Path(__file__).resolve().parent.parent
    matches_dir: Path = Path(__file__).resolve().parent / "matches"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
# v3.3 — apply the Admin-Panel / Optuna runtime weights on startup.
try:
    settings.reload_runtime_weights()
except Exception:
    # Never block startup on a malformed artifact; defaults stay.
    pass
# v3.6 — apply Admin-Panel runtime flags (live/mock toggles, goal_model etc.).
try:
    settings.reload_runtime_flags()
except Exception:
    pass
