# Observability: metrics and alerts

The API exposes Prometheus metrics at `GET /metrics`. This note lists what is measured, how to
scrape it, and which alerts I would set and why. The rules themselves live in
[`ops/prometheus/alerts.yml`](../ops/prometheus/alerts.yml) and pass `promtool check rules`.

## See the metrics: Prometheus + Grafana

The industry-standard open-source setup is **Prometheus** (scrapes and stores metrics, evaluates
alert rules) with **Grafana** (dashboards) and **Alertmanager** (routes firing alerts to Slack,
PagerDuty or email). Hosted alternatives work on the same idea: Grafana Cloud, Datadog, New Relic,
CloudWatch, often fed through OpenTelemetry. This repo ships the first two as an opt-in Docker
Compose profile:

```bash
docker compose --profile monitoring up -d
```

| Tool | URL | What you see |
|---|---|---|
| Grafana | http://localhost:3000 | the **Northwind Helpdesk** dashboard opens as the home page (read-only without login; admin login from `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`) |
| Prometheus | http://localhost:9090 | raw queries, scrape targets (*Status → Targets*) and alert states (*Alerts*: inactive, pending or firing) |
| Raw metrics | http://localhost:8000/metrics | the text Prometheus scrapes every 15 s |

Everything is provisioned from files, so nothing is clicked together by hand:
[`ops/grafana/provisioning/`](../ops/grafana/provisioning) connects the Prometheus data source and
loads [`ops/grafana/dashboards/helpdesk.json`](../ops/grafana/dashboards/helpdesk.json).

The dashboard shows: questions answered, p95 `/ask` latency, refusal rate, index freshness, cache
hit ratio, thumbs-up rate, estimated LLM cost and 5xx rate as headline numbers, plus graphs of
requests by route and status, p50/p95 latency, answer outcomes, cache lookups, LLM tokens and
feedback.

Alertmanager is not included: Prometheus evaluates the rules and shows their state on its *Alerts*
page, but routing notifications needs a real destination (Slack webhook, PagerDuty key), which is
environment-specific.

## Metrics

| Metric | Type | Labels | What it tells you |
|---|---|---|---|
| `helpdesk_http_requests_total` | counter | `method`, `route`, `status` | traffic and error rate per endpoint |
| `helpdesk_http_request_duration_seconds` | histogram | `method`, `route` | latency; p50/p95 via `histogram_quantile` |
| `helpdesk_answers_total` | counter | `outcome` = `answered` / `refused` / `llm_error` | answer quality and provider health |
| `helpdesk_llm_tokens_total` | counter | `kind` = `prompt` / `completion` | token usage |
| `helpdesk_llm_cost_usd_total` | counter | | estimated spend (0 with the mock) |
| `helpdesk_feedback_total` | counter | `rating` = `up` / `down` | user satisfaction |
| `helpdesk_cache_requests_total` | counter | `result` = `hit` / `miss` / `error` | answer-cache effectiveness and health |
| `helpdesk_index_loaded` | gauge | | 1 if a search index is loaded |
| `helpdesk_index_stale` | gauge | | 1 if the index is older than the policy documents |

Design choices:
- **Route templates, not raw URLs.** The `route` label is the matched route (`/ask`); anything unmatched becomes `unmatched`. Raw paths would let any client create unlimited time series.
- **Latency buckets cover both modes:** 10 ms to 10 s. The mock answers in tens of milliseconds and a hosted LLM in 1–4 s; without the upper buckets every real-LLM request would land in `+Inf`.
- **The index gauges are computed at scrape time**, so a policy edited while the service runs shows up without waiting for traffic.
- **`/metrics` has no API key**, like `/health`. It reveals counts, not content, but it should still only be reachable from the internal network (Prometheus scrapes `api:8000` inside Docker; the example config is [`ops/prometheus/prometheus.yml`](../ops/prometheus/prometheus.yml)).

## Alerts I would set

`page` means wake someone up; `ticket` means look at it during working hours.

| Alert | Condition | Severity | Why |
|---|---|---|---|
| `HelpdeskDown` | scrape target down for 2 min | page | nobody can get answers |
| `HelpdeskIndexNotLoaded` | `helpdesk_index_loaded == 0` for 5 min | page | every `/ask` returns 503 |
| `HelpdeskHighErrorRate` | over 5% of `/ask` requests are 5xx for 10 min | page | broken service, even if the process is up |
| `HelpdeskLLMProviderFailing` | over 10% of answers are `llm_error` for 5 min | page | provider outage, revoked key or retired model; the client already retries 429/5xx, so these are persistent failures |
| `HelpdeskSlowAnswers` | p95 `/ask` latency over 5 s for 15 min | ticket | Groq measured about 1.3 s p50 and up to about 4 s; sustained p95 above 5 s means a degraded provider or rate limiting |
| `HelpdeskIndexStale` | `helpdesk_index_stale == 1` for 15 min | ticket | answers may cite outdated policy (the debugging drill's root cause); auto-rebuild failed or is turned off |
| `HelpdeskRefusalSpike` | over 40% "I don't know" for 30 min, with real traffic | ticket | retrieval regression, or staff asking about topics the policies do not cover |
| `HelpdeskThumbsDownSpike` | over 30% thumbs-down in a day, with at least 20 ratings | ticket | answer quality problem that metrics alone would miss |
| `HelpdeskLLMSpendHigh` | estimated spend above $5 per day | ticket | runaway usage; at about $0.0001 per question that is roughly 50,000 questions |
| `HelpdeskAuthFailures` | sustained 401s (over 0.5/s for 10 min) | ticket | a frontend shipped with the wrong key, or someone guessing keys |
| `HelpdeskCacheErrors` | any cache errors for 10 min | ticket | Redis is down; requests still work but slower and at full LLM cost |

Thresholds are starting points from this project's measurements, to be tuned once real traffic exists. The ratio alerts include minimum-volume conditions so a quiet night with two questions cannot fire them.

## Validate the rules

```bash
docker run --rm -v "$PWD/ops/prometheus:/etc/prom" --entrypoint promtool \
  prom/prometheus:v2.55.1 check rules /etc/prom/alerts.yml
```

On Git Bash for Windows, prefix the command with `MSYS_NO_PATHCONV=1` so the container path is not rewritten.

## Not covered

Logs are printed to stdout (`docker compose logs api`), with one line per `/ask` naming the retrieved chunks and scores. There is no log shipping, tracing or Alertmanager here; in production I would ship logs to a central store (for example Loki next to Grafana), add OpenTelemetry tracing across retrieval and the LLM call, and route alerts through Alertmanager.
