"""Tests for the optional Redis answer cache (fakeredis: no Redis server needed)."""

from collections.abc import Iterator
from typing import Any

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.cache import NullCache, RedisCache, cache_key, make_cache, normalise_question
from app.config import Settings
from app.main import create_app
from app.models import QueryLog

HEADERS = {"X-API-Key": "test-key"}
QUESTION = {"question": "What is the hotel limit per night?"}


# --- keys -------------------------------------------------------------------------------


def test_normalisation_ignores_case_spacing_and_trailing_punctuation() -> None:
    assert normalise_question("  What is the HOTEL   limit? ") == "what is the hotel limit"
    assert cache_key("What is the hotel limit?", v=1) == cache_key("what is the hotel limit", v=1)


def test_different_wording_or_context_gives_a_different_key() -> None:
    assert cache_key("hotel limit", index_built_at="t1") != cache_key(
        "car limit", index_built_at="t1"
    )
    # A rebuilt index (new built_at) must never reuse answers computed from the old one.
    assert cache_key("hotel limit", index_built_at="t1") != cache_key(
        "hotel limit", index_built_at="t2"
    )
    assert cache_key("hotel limit", model="a") != cache_key("hotel limit", model="b")


# --- backends -----------------------------------------------------------------------------


def test_redis_cache_round_trip_with_ttl() -> None:
    client = fakeredis.FakeRedis()
    cache = RedisCache("redis://unused", ttl_seconds=60, client=client)
    assert cache.get("k") is None
    cache.set("k", {"answer": "24 days", "sources": []})
    assert cache.get("k") == {"answer": "24 days", "sources": []}
    assert 0 < client.ttl("k") <= 60


def test_cache_is_off_without_a_url() -> None:
    assert isinstance(make_cache("", 60), NullCache)
    assert isinstance(make_cache("redis://localhost:6379/0", 60), RedisCache)


# --- the API ------------------------------------------------------------------------------


class BrokenCache:
    """A cache whose Redis is down."""

    enabled = True

    def get(self, key: str) -> Any:
        raise redis.ConnectionError("redis down")

    def set(self, key: str, value: Any) -> None:
        raise redis.ConnectionError("redis down")


@pytest.fixture
def cached_client(settings: Settings) -> Iterator[tuple[TestClient, list[str]]]:
    """App with a fakeredis cache; also returns a list that records every real LLM call."""
    app = create_app(settings)
    app.state.cache = RedisCache("redis://unused", 60, client=fakeredis.FakeRedis())
    calls: list[str] = []
    with TestClient(app) as client:
        llm = app.state.llm
        original = llm.generate

        def counting_generate(question: str, chunks: Any) -> Any:
            calls.append(question)
            return original(question, chunks)

        llm.generate = counting_generate
        yield client, calls


def test_repeat_question_is_served_from_cache_without_calling_the_llm(cached_client) -> None:
    client, calls = cached_client
    first = client.post("/ask", json=QUESTION, headers=HEADERS).json()
    second = client.post(
        "/ask", json={"question": "what is the hotel limit per night"}, headers=HEADERS
    ).json()
    assert (first["cached"], second["cached"]) == (False, True)
    assert second["answer"] == first["answer"] and second["sources"] == first["sources"]
    assert len(calls) == 1  # the second request never reached the LLM
    assert second["query_id"] != first["query_id"]  # every request is still logged


def test_cache_hits_are_logged_as_cache_with_zero_cost(cached_client) -> None:
    client, _ = cached_client
    client.post("/ask", json=QUESTION, headers=HEADERS)
    client.post("/ask", json=QUESTION, headers=HEADERS)
    with client.app.state.session_factory() as db:
        rows = db.scalars(select(QueryLog).order_by(QueryLog.id)).all()
    assert [r.provider for r in rows] == ["mock", "cache"]
    assert rows[1].est_cost_usd == 0.0 and rows[1].prompt_tokens == 0


def test_rebuilt_index_invalidates_cached_answers(cached_client) -> None:
    client, calls = cached_client
    client.post("/ask", json=QUESTION, headers=HEADERS)
    client.app.state.retriever.metadata["built_at"] = "rebuilt-later"
    assert client.post("/ask", json=QUESTION, headers=HEADERS).json()["cached"] is False
    assert len(calls) == 2


def test_cache_metrics_count_hits_and_misses(cached_client) -> None:
    client, _ = cached_client
    client.post("/ask", json=QUESTION, headers=HEADERS)
    client.post("/ask", json=QUESTION, headers=HEADERS)
    text = client.get("/metrics").text
    assert 'helpdesk_cache_requests_total{result="miss"} 1.0' in text
    assert 'helpdesk_cache_requests_total{result="hit"} 1.0' in text


def test_redis_outage_still_answers_and_counts_errors(settings: Settings) -> None:
    app = create_app(settings)
    app.state.cache = BrokenCache()
    with TestClient(app) as client:
        resp = client.post("/ask", json=QUESTION, headers=HEADERS)
        assert resp.status_code == 200 and "$180" in resp.json()["answer"]
        assert resp.json()["cached"] is False
        text = client.get("/metrics").text
    assert 'helpdesk_cache_requests_total{result="error"} 2.0' in text  # failed get + failed set


def test_without_a_cache_nothing_is_cached(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        client.post("/ask", json=QUESTION, headers=HEADERS)
        assert client.post("/ask", json=QUESTION, headers=HEADERS).json()["cached"] is False
