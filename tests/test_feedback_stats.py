"""Tests for query logging, /feedback and /stats."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from app.config import Settings
from app.main import create_app
from app.models import Feedback, QueryLog

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def ask(client: TestClient, question: str = "What is the hotel limit per night?") -> dict:
    resp = client.post("/ask", json={"question": question}, headers=HEADERS)
    assert resp.status_code == 200
    return resp.json()


def test_ask_logs_a_query_row_and_returns_its_id(client: TestClient) -> None:
    body = ask(client)
    assert isinstance(body["query_id"], int)
    with client.app.state.session_factory() as db:
        row = db.get(QueryLog, body["query_id"])
    assert row is not None
    assert row.question == "What is the hotel limit per night?"
    assert row.answer == body["answer"]
    assert row.sources[0]["doc"] == "travel"
    assert row.top_score == pytest.approx(body["sources"][0]["score"], abs=1e-3)
    assert row.provider == "mock" and row.est_cost_usd == 0.0
    assert row.prompt_tokens > 0 and row.completion_tokens > 0
    assert row.created_at is not None


def test_feedback_is_stored(client: TestClient) -> None:
    qid = ask(client)["query_id"]
    resp = client.post("/feedback", json={"query_id": qid, "rating": "up"}, headers=HEADERS)
    assert resp.status_code == 200 and resp.json() == {"query_id": qid, "rating": "up"}
    with client.app.state.session_factory() as db:
        assert db.scalars(select(Feedback.rating)).all() == ["up"]


def test_revoting_replaces_the_rating(client: TestClient) -> None:
    qid = ask(client)["query_id"]
    client.post("/feedback", json={"query_id": qid, "rating": "up"}, headers=HEADERS)
    client.post("/feedback", json={"query_id": qid, "rating": "down"}, headers=HEADERS)
    with client.app.state.session_factory() as db:
        assert db.scalars(select(Feedback.rating)).all() == ["down"]


def test_feedback_for_unknown_query_is_404(client: TestClient) -> None:
    resp = client.post("/feedback", json={"query_id": 999, "rating": "up"}, headers=HEADERS)
    assert resp.status_code == 404


def test_feedback_rejects_invalid_rating(client: TestClient) -> None:
    qid = ask(client)["query_id"]
    resp = client.post("/feedback", json={"query_id": qid, "rating": "meh"}, headers=HEADERS)
    assert resp.status_code == 422


def test_feedback_and_stats_require_api_key(client: TestClient) -> None:
    assert client.post("/feedback", json={"query_id": 1, "rating": "up"}).status_code == 401
    assert client.get("/stats").status_code == 401


def test_stats_with_no_data(client: TestClient) -> None:
    resp = client.get("/stats", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {
        "count": 0,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "total_est_cost_usd": 0.0,
        "avg_est_cost_usd": None,
        "feedback_count": 0,
        "thumbs_up_rate": None,
    }


def test_stats_aggregates_queries_and_feedback(client: TestClient) -> None:
    ids = [ask(client)["query_id"] for _ in range(4)]
    client.post("/feedback", json={"query_id": ids[0], "rating": "up"}, headers=HEADERS)
    client.post("/feedback", json={"query_id": ids[1], "rating": "up"}, headers=HEADERS)
    client.post("/feedback", json={"query_id": ids[2], "rating": "down"}, headers=HEADERS)

    stats = client.get("/stats", headers=HEADERS).json()
    assert stats["count"] == 4
    assert stats["feedback_count"] == 3
    assert stats["thumbs_up_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert 0 <= stats["p50_latency_ms"] <= stats["p95_latency_ms"]
    assert stats["total_est_cost_usd"] == 0.0 and stats["avg_est_cost_usd"] == 0.0


def test_stats_use_known_latencies_and_costs(client: TestClient) -> None:
    with client.app.state.session_factory() as db:
        for latency, cost in [(10.0, 0.01), (20.0, 0.03), (30.0, 0.02), (40.0, 0.04)]:
            db.add(
                QueryLog(
                    question="q",
                    answer="a",
                    sources=[],
                    top_score=0.5,
                    latency_ms=latency,
                    provider="openai",
                    prompt_tokens=1,
                    completion_tokens=1,
                    est_cost_usd=cost,
                )
            )
        db.commit()
    stats = client.get("/stats", headers=HEADERS).json()
    assert stats["p50_latency_ms"] == 25.0
    assert stats["p95_latency_ms"] == pytest.approx(38.5)
    assert stats["total_est_cost_usd"] == pytest.approx(0.10)
    assert stats["avg_est_cost_usd"] == pytest.approx(0.025)


def test_expected_indexes_exist(client: TestClient) -> None:
    inspector = inspect(client.app.state.engine)
    queries = {i["name"]: i["column_names"] for i in inspector.get_indexes("queries")}
    feedback = {i["name"]: i["column_names"] for i in inspector.get_indexes("feedback")}
    assert queries["ix_queries_created_at"] == ["created_at"]
    assert feedback["ix_feedback_query_id"] == ["query_id"]


def test_db_failure_does_not_break_ask(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy.exc import OperationalError

    def broken_factory() -> None:
        raise OperationalError("insert", {}, Exception("db down"))

    monkeypatch.setattr(client.app.state, "session_factory", broken_factory)
    body = ask(client)
    assert body["query_id"] is None and "$180" in body["answer"]
