"""Prometheus metrics for the helpdesk API (served at GET /metrics).

Each app instance owns its own registry, so tests that create several apps never collide on
metric names, and nothing leaks into prometheus_client's global default registry.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Buckets span the two regimes this service runs in: ~10-50 ms with the offline mock and
# ~1-4 s with a hosted LLM. Without the upper buckets every real-LLM request would land in
# +Inf and p95 could not be estimated.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    """All metrics for one app instance."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "helpdesk_http_requests",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "helpdesk_http_request_duration_seconds",
            "End-to-end request handling time by method and route template.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.answers = Counter(
            "helpdesk_answers",
            'Outcomes of /ask: answered, refused ("I don\'t know") or llm_error.',
            ["outcome"],
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "helpdesk_llm_tokens",
            "LLM tokens used, by kind (prompt or completion).",
            ["kind"],
            registry=self.registry,
        )
        self.llm_cost = Counter(
            "helpdesk_llm_cost_usd",
            "Estimated LLM spend in US dollars.",
            registry=self.registry,
        )
        self.cache = Counter(
            "helpdesk_cache_requests",
            "Answer-cache lookups by result (hit, miss or error).",
            ["result"],
            registry=self.registry,
        )
        self.index_loaded = Gauge(
            "helpdesk_index_loaded",
            "1 if a search index is loaded, else 0.",
            registry=self.registry,
        )
        self.index_stale = Gauge(
            "helpdesk_index_stale",
            "1 if the loaded index is older than the policy documents, else 0.",
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        """Serialise every metric in the Prometheus text exposition format."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
