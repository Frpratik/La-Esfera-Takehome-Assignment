"""Tests for the Prometheus /metrics endpoint."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from app.config import Settings
from app.main import create_app

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def sample(client: TestClient, name: str, **labels: str) -> float:
    """Value of one sample in the /metrics output (0.0 if absent), parsed like Prometheus."""
    text = client.get("/metrics").text
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return 0.0


def test_metrics_endpoint_serves_prometheus_text_without_a_key(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "# TYPE helpdesk_http_requests_total counter" in resp.text
    assert "# TYPE helpdesk_http_request_duration_seconds histogram" in resp.text


def test_requests_are_counted_by_route_and_status(client: TestClient) -> None:
    client.post("/ask", json={"question": "hotel limit per night"}, headers=HEADERS)
    client.post("/ask", json={"question": "hotel limit per night"})  # no key -> 401
    assert sample(client, "helpdesk_http_requests_total", route="/ask", status="200") == 1
    assert sample(client, "helpdesk_http_requests_total", route="/ask", status="401") == 1


def test_latency_histogram_records_each_request(client: TestClient) -> None:
    for _ in range(3):
        client.post("/ask", json={"question": "annual leave days"}, headers=HEADERS)
    count = sample(client, "helpdesk_http_request_duration_seconds_count", route="/ask")
    total = sample(client, "helpdesk_http_request_duration_seconds_sum", route="/ask")
    assert count == 3 and total > 0


def test_unknown_paths_share_one_label(client: TestClient) -> None:
    """Raw URLs must not become labels, or random paths would create unbounded series."""
    client.get("/does-not-exist-1")
    client.get("/does-not-exist-2")
    assert sample(client, "helpdesk_http_requests_total", route="unmatched", status="404") == 2
    assert "does-not-exist" not in client.get("/metrics").text


def test_answer_outcomes_tokens_and_cost(client: TestClient) -> None:
    client.post("/ask", json={"question": "What is the hotel limit per night?"}, headers=HEADERS)
    client.post("/ask", json={"question": "zebra quantum xylophone"}, headers=HEADERS)
    assert sample(client, "helpdesk_answers_total", outcome="answered") == 1
    assert sample(client, "helpdesk_answers_total", outcome="refused") == 1
    assert sample(client, "helpdesk_llm_tokens_total", kind="prompt") > 0
    assert sample(client, "helpdesk_llm_cost_usd_total") == 0.0  # mock is free


def test_index_gauges_reflect_a_policy_edit(client: TestClient, policies_dir: Path) -> None:
    assert sample(client, "helpdesk_index_loaded") == 1
    assert sample(client, "helpdesk_index_stale") == 0
    path = policies_dir / "travel.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nNew rule.", encoding="utf-8")
    assert sample(client, "helpdesk_index_stale") == 1


def test_each_app_has_its_own_registry(settings: Settings) -> None:
    with TestClient(create_app(settings)) as first, TestClient(create_app(settings)) as second:
        first.post("/ask", json={"question": "hotel"}, headers=HEADERS)
        assert sample(first, "helpdesk_http_requests_total", route="/ask", status="200") == 1
        assert sample(second, "helpdesk_http_requests_total", route="/ask", status="200") == 0


def test_feedback_ratings_are_counted(client: TestClient) -> None:
    qid = client.post("/ask", json={"question": "hotel limit"}, headers=HEADERS).json()["query_id"]
    client.post("/feedback", json={"query_id": qid, "rating": "down"}, headers=HEADERS)
    assert sample(client, "helpdesk_feedback_total", rating="down") == 1
    assert sample(client, "helpdesk_feedback_total", rating="up") == 0


def test_known_label_values_start_at_zero(client: TestClient) -> None:
    """So Prometheus sees the 0 -> 1 step and rate()/increase() count the first event."""
    text = client.get("/metrics").text
    for line in (
        'helpdesk_answers_total{outcome="refused"} 0.0',
        'helpdesk_cache_requests_total{result="hit"} 0.0',
        'helpdesk_feedback_total{rating="down"} 0.0',
    ):
        assert line in text
