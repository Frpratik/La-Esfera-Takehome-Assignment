# Policy Helpdesk

An internal helpdesk assistant for **Northwind Facilities**, a mid-sized property management
firm. Staff ask questions such as "How many days notice do I need for annual leave?" or "Who
approves a laptop replacement?" and get answers grounded in 8 short policy documents. It is a
small RAG service with a chat UI, Docker setup, an evaluation harness and a debugging write-up.
All data is synthetic.

## Design note

### Components
| Component | Role |
|---|---|
| `data/policies/` | 8 markdown policy documents (synthetic) |
| `ingest/` | Chunks docs, embeds them, writes a FAISS index plus JSON metadata (doc, chunk_id, text, per-doc content hash, build timestamp) |
| `app/` | FastAPI service: `/ask`, `/feedback`, `/stats`, `/health`; retriever, `LLMClient` interface, prompt template, DB logging |
| `frontend/` | React + TypeScript + Vite chat UI served by nginx |
| `eval/` | 12-question eval: retrieval hit rate, answer correctness, refusal accuracy, latency, cost |
| `tests/` | pytest suite (chunker, API, auth, stats, stale index) |

Flow: question → embed → FAISS top-k → prompt (context only) → LLM → answer + sources → logged to DB.

### Tech choices and why
- **FastAPI + Pydantic settings**: typed request/response models and env-driven config.
- **all-MiniLM-L6-v2 + hashing fallback**: small, good enough for short policy text. The fallback embedder is deterministic, so the project runs offline and tests are stable.
- **FAISS + JSON metadata**: simple, no extra server. Per-document content hashes let us detect a stale index.
- **`LLMClient` interface (mock | openai-compatible)**: the default mock needs no network or key. One OpenAI-compatible client covers Groq and Ollama via `base_url`.
- **SQLAlchemy, Postgres / SQLite**: Postgres in compose, SQLite for local dev and tests, one code path.
- **Docker Compose**: `api`, `frontend` and `postgres` start from a copied `.env.example`.

### Main risks
- **Stale index**: policies change but the index doesn't. Mitigation: hash comparison at startup and in `/health`.
- **Retrieval confusion**: related docs (expenses vs travel) with similar wording. Mitigation: keep each fact in one policy (no restating another policy's topic), hybrid dense + BM25 retrieval, a top-1 document metric and a debug script (see [docs/DEBUGGING.md](docs/DEBUGGING.md)).
- **Hallucination**: the prompt restricts answers to context and requires an exact "I don't know" refusal. The eval measures refusal accuracy.
- **Offline embedder quality**: the hashing fallback is weaker than MiniLM, so eval numbers differ per embedder. Report which one was used.
- **Cost/latency drift**: tracked per query in the DB and exposed via `/stats`.

### 4-hour time split
| Phase | Time |
|---|---|
| 0 Plan & scaffold | 0:15 |
| 1 Synthetic data | 0:20 |
| 2 Ingestion | 0:30 |
| 3 Core RAG API | 0:40 |
| 4 Logging, feedback, stats | 0:30 |
| 5 Chat frontend | 0:35 |
| 6 Docker | 0:25 |
| 7 Evaluation | 0:20 |
| 8 Debugging drill + docs | 0:25 |

## Run it

Requires Docker with Compose v2. No API keys or internet are needed at runtime (mock LLM).

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:8080 |
| API | http://localhost:8000 (docs at `/docs`) |
| Postgres | internal only (`postgres:5432`) |

Try it:

```bash
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" -d '{"question":"What is the hotel limit per night?"}'
curl -s localhost:8000/stats -H "X-API-Key: change-me"
```

How the stack starts: Postgres becomes healthy, then the API starts (the FAISS index is already baked into the image; the entrypoint rebuilds it only if missing), then the frontend starts once the API reports healthy.

- **Real LLM:** set `LLM_PROVIDER=openai` and `LLM_API_KEY` in `.env`. The defaults point at Groq with `openai/gpt-oss-20b`; for Ollama set `LLM_BASE_URL=http://host.docker.internal:11434/v1` and its model name. Set `PRICE_PER_1K_PROMPT=0.000075` and `PRICE_PER_1K_COMPLETION=0.0003` (Groq's published gpt-oss-20b prices) for cost tracking, then `docker compose up -d --build`. The client retries rate limits (HTTP 429) with backoff.
- **Changing `API_KEY` or `PUBLIC_API_URL`:** the frontend bakes them in at build time, so rebuild it (`docker compose up -d --build frontend`).
- **Offline build:** the image downloads the MiniLM model at build time. If that fails, the build continues and the index uses the hash embedder (a warning is printed).
- **Reset data:** `docker compose down -v` also removes the Postgres volume.
- **Local dev without Docker:** see "Ingestion" and "Frontend (local dev)" below. Tests: `pip install -r requirements-dev.txt && pytest`.

## Ingestion

Build the index with `python -m ingest.build_index` (set `USE_HASH_EMBEDDER=true` to run fully offline).

**Chunking: 120 words per chunk, 30 words (25%) overlap.**
- *Why 120 words:* policy sections are short (about 40-60 words), so a chunk holds one to two topics. All-MiniLM-L6-v2 truncates at 256 word-pieces; 120 words is roughly 160 tokens, safely under the cap. Smaller chunks dilute the answer, larger ones mix topics and lower precision.
- *Why 30 words of overlap:* a fact cut by a chunk boundary still appears whole in the neighbouring chunk, provided the sentence is shorter than the overlap. More overlap would mostly add duplicate chunks.
- Both values are env-configurable (`CHUNK_SIZE`, `CHUNK_OVERLAP`) and recorded in the index metadata.
- Document hashes ignore CRLF vs LF, so a Windows checkout does not look stale in a Linux container.
- The index stores `built_at`, the embedder name, the chunking strategy and a SHA-256 per source document. At startup the API compares them with `data/policies`: a stale index is logged, reported by `/health` (`status: "stale"`), and rebuilt when `AUTO_REBUILD_INDEX=true` (the default). `python -m ingest.freshness` runs the same check and exits 1 when stale.

**Retrieval:** `RETRIEVAL_MODE=hybrid` (the default) fuses dense similarity with BM25 via Reciprocal Rank Fusion. BM25's exact-term matching helps separate near-duplicate policies; it is a safeguard, and the main fix for the drill's confusion was removing duplicated content (see [docs/DEBUGGING.md](docs/DEBUGGING.md)). `python -m app.debug "<question>"` prints the top-k chunks with cosine and BM25 scores.

## Frontend (local dev)

```bash
# terminal 1: API (offline hash embedder + mock LLM)
USE_HASH_EMBEDDER=true python -m ingest.build_index
API_KEY=demo-key python -m uvicorn app.main:create_app --factory --port 8000

# terminal 2: UI at http://localhost:5173
cd frontend && cp .env.example .env   # set VITE_API_KEY=demo-key
npm install && npm run dev
```

`VITE_API_BASE_URL` and `VITE_API_KEY` are baked into the JS bundle at build time, so the key is visible to anyone who can load the page. That is acceptable for this take-home; a real deployment would put auth behind a login or a server-side proxy. The API allows only the origins listed in `CORS_ORIGINS`.

## Evaluation

`eval/questions.json` holds 12 questions: 9 answerable (each with an expected document and fact keywords, covering all 8 policies) and 3 unanswerable. `python -m eval.run_eval` asks them through `/ask` (in-process, offline, throwaway database; or `--url` against a running service), prints a table and writes `eval/results.json`.

```bash
python -m ingest.build_index      # the index must exist
python -m eval.run_eval
```

| Metric | What it measures |
|---|---|
| Retrieval hit rate | expected document appears in the top-4 sources (answerable questions) |
| Top-1 document rate | expected document is the first source (catches near-duplicate competition the hit rate misses) |
| Answer correctness | answer is not a refusal and contains every expected fact (case-insensitive match) |
| Refusal accuracy | unanswerable questions that got exactly "I don't know" |
| False refusal rate | answerable questions wrongly refused |
| Latency / cost | average server-side `latency_ms`; cost is the change in `/stats` total divided by questions |

Results (top-k = 4, hybrid retrieval):

| Setup | Retrieval hit | Top-1 doc | Correctness | Refusal acc. | False refusals | Avg latency | Cost/query |
|---|---|---|---|---|---|---|---|
| MiniLM + mock LLM (`eval/results.json`) | 100% | 100% | 100% | 100% | 0% | 37 ms | $0 |
| MiniLM + **Groq `openai/gpt-oss-20b`** (`eval/results_groq.json`) | 100% | 100% | 100% | 100% | 0% | 1.8 s | $0.00010 |
| Hash fallback embedder + mock LLM | 100% | 88.9% | 77.8% | 100% | 22.2% | 2 ms | $0 |

A second set, `eval/regression_questions.json` (8 travel reimbursement and expense questions from the debugging drill), is run with `python -m eval.run_eval --questions eval/regression_questions.json`. With MiniLM: top-1 document 100%; correctness 100% with Groq and 75% with the mock. The two mock misses are a limit of its keyword matching, explained in [docs/DEBUGGING.md](docs/DEBUGGING.md). `--rescore <results.json>` re-applies the current scoring to a saved run without new API calls.

**How to read these numbers**
- The mock LLM is free, so its cost is $0 by construction. The Groq run used Groq's published gpt-oss-20b prices ($0.075 / $0.30 per 1M input/output tokens); the docs page shows no "as of" date, so treat the figure as an estimate. Real-LLM answers vary slightly between runs (temperature 0 is not a determinism guarantee), so its numbers are a snapshot, not a fixed baseline.
- The live run also found bugs the mock could not: a retired default model, no handling of rate limits, BM25 plural and determinism bugs, and a scorer that rejected correct answers written with markdown or narrow no-break spaces. All are fixed and listed in [docs/DEBUGGING.md](docs/DEBUGGING.md).
- The set is small and the mock answers by keyword overlap, so 100% shows the pipeline works end to end, not production quality. The two numbers are still informative: the hash fallback finds the right document every time but its lower similarity scores fall under the refusal threshold, so q03 and q08 are wrongly refused. That is why retrieval hit rate and answer correctness are reported separately.
- The unanswerable questions are caught by the score threshold (0.35 for MiniLM), tuned on observed scores, so a different embedder needs a different threshold.
- Latency on the first question includes warm-up; the in-process numbers exclude network time.
