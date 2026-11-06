# Policy Helpdesk

A small RAG service that answers employee questions from internal policy documents,
with a chat UI, Docker setup, an evaluation harness and a debugging write-up.

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
- **Retrieval confusion**: related docs (expenses vs travel) with similar wording. Mitigation: chunk sizing, overlap, and top-k inspection tooling.
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

- **Real LLM:** set `LLM_PROVIDER=openai`, `LLM_API_KEY` and `LLM_MODEL` in `.env` (default base URL is Groq; point `LLM_BASE_URL` at Ollama, e.g. `http://host.docker.internal:11434/v1`), plus the `PRICE_PER_1K_*` values for cost tracking, then `docker compose up -d --build`.
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
- The index stores `built_at`, the embedder name, and a SHA-256 per source document, which later lets us detect a stale index.

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
