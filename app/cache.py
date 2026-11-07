"""Optional Redis cache for /ask answers to repeated questions.

Disabled unless CACHE_URL is set. A cache failure never fails a request: callers treat errors
as misses (see app/main.py) and the service keeps answering, just without the speed-up.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

import redis

KEY_PREFIX = "helpdesk:answer:"


class AnswerCache(Protocol):
    """Store of previously computed /ask payloads, keyed by :func:`cache_key`."""

    enabled: bool

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached payload, or None on a miss."""
        ...

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a payload (with the cache's TTL)."""
        ...


class NullCache:
    """Used when caching is off: every lookup misses and nothing is stored."""

    enabled = False

    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        return None


class RedisCache:
    """Redis-backed cache. JSON values, expiring after ``ttl_seconds``."""

    enabled = True

    def __init__(self, url: str, ttl_seconds: int, client: redis.Redis | None = None) -> None:
        # Short socket timeouts: if Redis hangs, a lookup must fail fast rather than add
        # seconds to every request. redis-py connects lazily, so this never blocks startup.
        self._client = client or redis.Redis.from_url(
            url, socket_timeout=0.25, socket_connect_timeout=0.25
        )
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> dict[str, Any] | None:
        raw = self._client.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._client.setex(key, self.ttl_seconds, json.dumps(value))

    def ping(self) -> bool:
        """True if Redis answers; used only to log a warning at startup."""
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False


def normalise_question(question: str) -> str:
    """Lowercase, collapse whitespace and drop trailing punctuation.

    So "What is the hotel limit?" and "what is the  hotel limit" share one cache entry,
    while any change in wording that could change the answer still gets its own entry.
    """
    text = re.sub(r"\s+", " ", question.strip().lower())
    return text.rstrip(" ?!.")


def cache_key(question: str, **context: Any) -> str:
    """Key for a question plus everything that could change its answer.

    ``context`` should include the index build time, embedder, retrieval settings and LLM
    provider/model. Including the index build time is what keeps the cache from
    reintroducing the stale-answer bug: after a policy edit the index is rebuilt, its
    ``built_at`` changes, and every old entry simply stops being looked up (then expires).
    """
    material = json.dumps(
        {"q": normalise_question(question), **context}, sort_keys=True, default=str
    )
    return KEY_PREFIX + hashlib.sha256(material.encode("utf-8")).hexdigest()


def make_cache(url: str, ttl_seconds: int) -> AnswerCache:
    """A RedisCache when ``url`` is set, otherwise the no-op NullCache."""
    return RedisCache(url, ttl_seconds) if url else NullCache()
