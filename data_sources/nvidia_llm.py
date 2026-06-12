"""NVIDIA build.nvidia.com LLM connector.

build.nvidia.com hosts OpenAI-compatible ``/v1/chat/completions`` endpoints for
open-weight models (Llama, Mistral, …). We talk to them via plain httpx so the
SDK isn't a required dependency — keeps the surface narrow and consistent with
every other connector inheriting :class:`BaseConnector`.

Two methods:

* ``score_sentiment(texts, home_code, away_code)``
    Returns aspect-level polarities (`attack`, `defence`, `morale`) per side
    plus an overall polarity / intensity / confidence triple. Uses a JSON-schema
    response format ("guided generation") so the model can't wander off-prompt.

* ``summarize_news(articles, home_code, away_code)``
    Compresses scraped news/forum posts into a structured digest the LLM-Meta
    layer (and the UI explainer in a later phase) can consume.

Failure modes degrade to the mock payload — no exception ever escapes into the
factor layer. Token budget is governed by ``settings.llm_request_budget_per_match``.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Sequence

import httpx
import structlog

from config.settings import settings
from data_sources.base import BaseConnector, FetchResult
from data_sources.mock import nvidia_llm_mock

log = structlog.get_logger("data_sources.nvidia_llm")

_LLM_TTL_S = 60 * 30.0   # 30 min — long enough to dedupe a re-crawl, short
                          # enough that the panel stays useful pre-kickoff.

_SENTIMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["home", "away", "model"],
    "properties": {
        "model": {"type": "string"},
        "samples": {"type": "integer"},
        "home": {
            "type": "object",
            "additionalProperties": False,
            "required": ["polarity", "intensity", "confidence"],
            "properties": {
                "polarity":   {"type": "number", "minimum": -1, "maximum": 1},
                "intensity":  {"type": "number", "minimum": 0,  "maximum": 1},
                "confidence": {"type": "number", "minimum": 0,  "maximum": 1},
                "aspects": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "attack":  {"type": "number", "minimum": -1, "maximum": 1},
                        "defence": {"type": "number", "minimum": -1, "maximum": 1},
                        "morale":  {"type": "number", "minimum": -1, "maximum": 1},
                    },
                },
            },
        },
        "away": {"$ref": "#/properties/home"},
    },
}


class NvidiaLlmConnector(BaseConnector):
    """OpenAI-compatible LLM client targeting build.nvidia.com."""

    connector_name = "nvidia_llm"

    def _default_headers(self) -> dict[str, str]:
        h = {"User-Agent": settings.reddit_user_agent, "Accept": "application/json"}
        if settings.nvidia_api_key:
            h["Authorization"] = f"Bearer {settings.nvidia_api_key}"
        return h

    @property
    def available(self) -> bool:
        return bool(settings.nvidia_api_key) and settings.use_nvidia_llm

    async def score_sentiment(
        self,
        texts: Sequence[str],
        home_code: str,
        away_code: str,
    ) -> FetchResult:
        """Aspect-sentiment scoring over a small bag of posts.

        ``texts`` is already pre-filtered to the top-N posts per Reddit tier;
        the connector enforces a hard list length cap so the chat-completions
        token budget can't blow up.
        """
        home_code = home_code.upper()
        away_code = away_code.upper()
        if not self.available:
            return FetchResult(
                nvidia_llm_mock.aspect_sentiment(home_code, away_code, len(texts)),
                "mock", None, "mock",
            )

        # Trim hard so a runaway crawl can't push the prompt past free-tier limits.
        max_total = max(1, settings.llm_max_posts_per_tier * 3)
        sample = list(texts)[:max_total]
        if not sample:
            return FetchResult(
                nvidia_llm_mock.aspect_sentiment(home_code, away_code, 0),
                "mock", None, "mock",
            )

        prompt = self._build_sentiment_prompt(sample, home_code, away_code)
        payload = {
            "model": settings.nvidia_llm_model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            "temperature": settings.llm_temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "AspectSentiment", "schema": _SENTIMENT_SCHEMA},
            },
            "max_tokens": 512,
        }

        try:
            data = await self._post_chat(payload)
        except Exception as exc:
            log.warning("nvidia_llm_call_failed", error=str(exc))
            return FetchResult(
                nvidia_llm_mock.aspect_sentiment(home_code, away_code, len(sample)),
                "error", None, "nvidia_llm",
            )

        try:
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:
            log.warning("nvidia_llm_parse_failed", error=str(exc))
            return FetchResult(
                nvidia_llm_mock.aspect_sentiment(home_code, away_code, len(sample)),
                "error", None, "nvidia_llm",
            )
        parsed.setdefault("samples", len(sample))
        return FetchResult(parsed, "live", None, self.connector_name)

    async def summarize_news(
        self,
        articles: Sequence[str],
        home_code: str,
        away_code: str,
    ) -> FetchResult:
        """Compress raw news snippets into a structured digest.

        Output shape:
            {"home_issues": [str], "away_issues": [str], "shared": [str]}
        Used by the InjuryNewsFactor when no structured FotMob/SofaScore feed
        is available; cheaper than tokenising every article through spaCy.
        """
        home_code = home_code.upper()
        away_code = away_code.upper()
        if not self.available or not articles:
            return FetchResult(
                {"home_issues": [], "away_issues": [], "shared": []},
                "mock" if not self.available else "live",
                None, self.connector_name,
            )

        budget = max(1, settings.llm_request_budget_per_match)
        snippets = list(articles)[: budget * 6]
        prompt_user = (
            f"You receive sports news snippets for an upcoming match between "
            f"{home_code} (home) and {away_code} (away). Extract injury, suspension, "
            f"and tactical issues that materially affect the match. Output strict JSON "
            f"{{home_issues:[...], away_issues:[...], shared:[...]}}.\n\n"
            + "\n---\n".join(snippets)
        )
        payload = {
            "model": settings.nvidia_llm_model,
            "messages": [
                {"role": "system", "content": "You produce concise structured football news digests."},
                {"role": "user", "content": prompt_user},
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": 600,
        }

        try:
            data = await self._post_chat(payload)
        except Exception as exc:
            log.warning("nvidia_llm_summarize_failed", error=str(exc))
            return FetchResult({"home_issues": [], "away_issues": [], "shared": []},
                               "error", None, self.connector_name)

        try:
            raw = data["choices"][0]["message"]["content"]
            digest = json.loads(raw)
        except Exception:
            digest = {"home_issues": [], "away_issues": [], "shared": []}
        return FetchResult(digest, "live", None, self.connector_name)

    # ── internals ────────────────────────────────────────────────────────────

    def _build_sentiment_prompt(
        self, sample: list[str], home: str, away: str,
    ) -> dict[str, str]:
        joined = "\n".join(f"- {t[:500]}" for t in sample if t)
        system = (
            "You score football fan sentiment from Reddit posts. "
            "Return polarity in [-1,1], intensity (excitement) in [0,1], "
            "and confidence in [0,1] separately for home and away. Decompose "
            "into three aspects: attack, defence, morale (each in [-1,1]). "
            "Be calibrated — most match threads are mildly mixed, not extreme."
        )
        user = (
            f"Home: {home}\nAway: {away}\n\nPosts:\n{joined}\n\n"
            "Respond as JSON matching the provided schema."
        )
        return {"system": system, "user": user}

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{settings.nvidia_llm_base_url.rstrip('/')}/chat/completions"
        client = await self._get_client()
        # Honour an instance-wide budget so a concurrent crawl can't run away.
        attempts = max(1, settings.datasource_retry_attempts)
        backoff = settings.datasource_retry_backoff_s
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.post(url, json=payload, timeout=settings.datasource_http_timeout_s)
            except httpx.HTTPError as exc:
                log.warning("nvidia_llm_http_error", attempt=attempt, error=str(exc))
                if attempt == attempts:
                    raise
                await asyncio.sleep(backoff * attempt)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                raise RuntimeError(f"nvidia_llm auth failed status={resp.status_code}")
            if resp.status_code == 429:
                # Honour Retry-After if present, else exponential backoff.
                retry_after = float(resp.headers.get("Retry-After", backoff * attempt))
                log.warning("nvidia_llm_rate_limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
                continue
            if 500 <= resp.status_code < 600 and attempt < attempts:
                await asyncio.sleep(backoff * attempt)
                continue
            raise RuntimeError(f"nvidia_llm status={resp.status_code} body={resp.text[:200]}")
        raise RuntimeError("nvidia_llm exhausted retries")


__all__ = ["NvidiaLlmConnector"]
