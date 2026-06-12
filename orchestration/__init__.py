"""v3.3 — optional Prefect orchestration layer.

Loaded only when ``settings.use_prefect`` is true. The APScheduler path in
``services/scheduled_jobs.py`` stays the default; Prefect adds DAG visibility,
retry policies, and a UI for the user who wants production-grade orchestration.
"""
