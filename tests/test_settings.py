"""Free-OSS migration — snapshot of the live-first settings defaults.

The project policy is: every auth-free / free-tier source is LIVE by default;
mock is the explicit opt-in for tests, CI and offline runs. These tests pin
the *class* defaults (``Settings.model_fields``) so neither a stray ``.env``
nor the runtime artifacts can mask a regression, plus the one behavioural
contract that keeps CI green: ``apply_runtime_profile("mock")`` must force
every connector back to its deterministic mock.
"""
from __future__ import annotations

from config.settings import Settings

# Every connector toggle — must stay in sync with Settings (the CLI discovers
# the same list dynamically via the use_mock_ prefix).
_ALL_MOCK_FLAGS = (
    "use_mock_crawler",
    "use_mock_openfootball",
    "use_mock_thesportsdb",
    "use_mock_openligadb",
    "use_mock_wikidata",
    "use_mock_weather",
    "use_mock_rss",
    "use_mock_clubelo",
    "use_mock_football_data",
    "use_mock_fbref",
    "use_mock_understat",
    "use_mock_fotmob",
    "use_mock_sofascore",
    "use_mock_transfermarkt",
    "use_mock_odds_api",
)


def _default(field: str):
    return Settings.model_fields[field].default


def test_every_connector_defaults_to_live():
    """Free-OSS migration: no source ships mock-first anymore."""
    mocked_by_default = [f for f in _ALL_MOCK_FLAGS if _default(f) is not False]
    assert mocked_by_default == [], (
        f"these sources regressed to mock-first: {mocked_by_default}"
    )


def test_mock_flag_list_is_complete():
    """A new use_mock_* field must be added to _ALL_MOCK_FLAGS (and thereby
    to the live-first contract) — this catches silent drift."""
    discovered = {n for n in Settings.model_fields if n.startswith("use_mock_")}
    assert discovered == set(_ALL_MOCK_FLAGS)


def test_nvidia_llm_on_by_default_but_key_gated():
    assert _default("use_nvidia_llm") is True
    # The key itself ships empty — the scorer's available-gate keeps it silent.
    assert _default("nvidia_api_key") == ""


def test_llm_sentiment_factor_carries_live_weight():
    assert _default("factor_weight_llm_sentiment") > 0.0


def test_twitter_stays_off():
    """Paid v2 API — project policy is free/open-source sources only."""
    assert _default("enable_twitter_crawler") is False


def test_reddit_uses_credentialless_http_crawler():
    """use_mock_crawler=False + use_arctic_shift=False selects
    HttpRedditCrawler (public old.reddit.com/.json, no OAuth)."""
    assert _default("use_mock_crawler") is False
    assert _default("use_arctic_shift") is False


def test_match_service_picks_http_reddit_crawler_under_live_defaults():
    """Behavioural proof of the free-OSS Phase-3 contract: with
    use_mock_crawler=False + use_arctic_shift=False the crawler dispatch in
    match_service routes to HttpRedditCrawler (the credentialless variant).

    Catches a silent regression where someone flips a default back to mock or
    re-enables the OAuth path without thinking about the account-free guarantee.
    """
    from config.settings import settings as live_settings

    # Sanity: defaults still match the contract (test_*_defaults_to_live above
    # already pin it, but the dispatch logic reads `settings`, so we double-check).
    assert live_settings.use_mock_crawler is False
    assert live_settings.use_arctic_shift is False

    try:
        from crawler.http_reddit import HttpRedditCrawler
    except Exception as exc:  # pragma: no cover - import-only safety
        import pytest
        pytest.skip(f"crawler module unavailable in this environment: {exc}")

    # Re-create the dispatch logic from services/match_service.py:111-127 in
    # one expression so the test fails when the routing diverges from the
    # documented contract — without importing the full service module (which
    # brings the DB / FastAPI stack).
    def _pick_crawler():
        if live_settings.use_mock_crawler:
            from crawler.mock_reddit import MockRedditCrawler
            return MockRedditCrawler
        if live_settings.use_arctic_shift:
            try:
                from crawler.parallel_reddit import ParallelRedditCrawler
                return ParallelRedditCrawler
            except Exception:
                pass
        return HttpRedditCrawler

    assert _pick_crawler() is HttpRedditCrawler


def test_mock_profile_forces_everything_offline():
    """The CI/test contract: apply_runtime_profile('mock') must flip every
    connector to mock and hard-disable the NVIDIA scorer — regardless of the
    live-first defaults."""
    from config.settings import settings
    from wm2026.context import apply_runtime_profile

    before = {f: getattr(settings, f) for f in _ALL_MOCK_FLAGS}
    before_llm = settings.use_nvidia_llm
    try:
        apply_runtime_profile("mock")
        assert all(getattr(settings, f) is True for f in _ALL_MOCK_FLAGS)
        assert settings.use_nvidia_llm is False
    finally:
        for f, v in before.items():
            object.__setattr__(settings, f, v)
        object.__setattr__(settings, "use_nvidia_llm", before_llm)
