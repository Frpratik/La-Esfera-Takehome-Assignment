"""API tests: /ask with the mock LLM, authentication and /health."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # The context manager runs the app lifespan, which loads the index.
    with TestClient(create_app(settings)) as c:
        yield c


def test_ask_returns_answer_sources_and_latency(client: TestClient) -> None:
    resp = client.post(
        "/ask", json={"question": "What is the hotel limit per night?"}, headers=HEADERS
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "$180" in body["answer"]
    assert body["sources"][0]["doc"] == "travel"
    assert set(body["sources"][0]) == {"doc", "chunk_id", "score"}
    assert body["latency_ms"] >= 0


def test_ask_unanswerable_question_gets_refusal(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "zebra quantum xylophone"}, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["answer"] == "I don't know"


def test_ask_rejects_empty_question(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "   "}, headers=HEADERS)
    assert resp.status_code == 422


def test_ask_without_key_is_401(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hotel"}).status_code == 401


def test_ask_with_wrong_key_is_401(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "hotel"}, headers={"X-API-Key": "nope"})
    assert resp.status_code == 401


def test_health_needs_no_key_and_reports_index(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["chunks"] > 0 and body["embedder"] == "hash-384"


def test_missing_index_gives_503_and_degraded_health(settings: Settings, tmp_path) -> None:
    broken = settings.model_copy(update={"index_dir": tmp_path / "nothing"})
    with TestClient(create_app(broken)) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/ask", json={"question": "hotel"}, headers=HEADERS).status_code == 503
