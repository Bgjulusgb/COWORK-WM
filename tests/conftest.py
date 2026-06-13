"""Shared pytest fixtures for the bare-pytest suites.

Deliberately tiny — the suites run on plain ``pytest`` (no pytest-asyncio) by
design (see CLAUDE.md), so this only wires up cross-test isolation that every
suite needs.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _clear_inmemory_cache():
    """Reset the process-global in-memory data cache between tests.

    ``utils.cache.cache`` holds the orchestrator's per-match snapshot
    (``orch:<match_id>`` — which now also carries ``live_odds``) plus every
    connector's per-call payload. Because it is a module-global singleton, two
    tests that predict the *same* match id would otherwise share a snapshot:
    the second test gets a cache hit and silently bypasses any monkeypatched
    connector. Clearing before each test keeps them independent (and matches
    the fresh-process assumption the CI list already relies on).
    """
    from utils.cache import cache

    asyncio.run(cache.invalidate())
    yield
    asyncio.run(cache.invalidate())


@pytest.fixture(autouse=True)
def _restore_live_first_settings():
    """Snapshot + restore the live-first Settings defaults around each test.

    Any test that runs the pipeline with ``mode="mock"`` triggers
    ``apply_runtime_profile("mock")``, which flips every ``use_mock_*`` flag
    and ``use_nvidia_llm`` on the global Settings singleton (via
    ``object.__setattr__`` — bypassing Pydantic guards). That mutation leaks
    into the next test, silently breaking anything that checks the live-first
    defaults (e.g. the Phase-1 contract tests). Snapshotting the toggled
    fields and restoring them after each test gives every test the same fresh
    Settings starting point that an isolated process would have.

    Same problem at the env-var layer: ``wm2026 predict`` calls
    ``_seed_source_toggles`` which writes ``USE_MOCK_*=true`` env vars
    (via ``os.environ.setdefault`` for mock mode, plain assignment for
    selective live-sources). Those leak into ``subprocess.run`` calls in
    later tests (e.g. ``test_doctor`` invokes the CLI in a child process
    that inherits the parent env). We snapshot the USE_MOCK_* slice of
    ``os.environ`` too and restore it.
    """
    import os

    from config.settings import settings
    from wm2026.context import _MOCK_FLAGS

    tracked = (*_MOCK_FLAGS, "use_nvidia_llm")
    before = {f: getattr(settings, f) for f in tracked if hasattr(settings, f)}
    env_before = {k: v for k, v in os.environ.items() if k.startswith("USE_MOCK_")}
    env_keys_before = {k for k in os.environ if k.startswith("USE_MOCK_")}
    try:
        yield
    finally:
        for f, v in before.items():
            object.__setattr__(settings, f, v)
        # Drop any USE_MOCK_* env added during the test, restore changed ones.
        for k in {k for k in os.environ if k.startswith("USE_MOCK_")} - env_keys_before:
            del os.environ[k]
        for k, v in env_before.items():
            os.environ[k] = v
