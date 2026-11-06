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
