"""FastAPI application: /ask, /health.

Run with: uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.llm import LLMError, MockLLMClient, get_llm_client
from app.retriever import Retriever
from app.schemas import AskRequest, AskResponse, Source

logger = logging.getLogger("helpdesk")


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests whose X-API-Key header is missing or wrong (401)."""
    expected = request.app.state.settings.api_key
    # compare_digest avoids leaking the key length/prefix through response timing.
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _load_retriever(settings: Settings) -> Retriever | None:
    """Load the index, or return None (service starts degraded) if it isn't available."""
    try:
        return Retriever.from_dir(settings.index_dir, settings.top_k)
    except Exception as exc:  # missing/corrupt index, embedder failure...
        logger.warning("index not loaded from %s: %s", settings.index_dir, exc)
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Takes explicit settings so tests can inject their own."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        retriever = _load_retriever(settings)
        llm = get_llm_client(settings)
        # Hash-embedder cosine scores sit on a lower scale than MiniLM's, so the mock's
        # refusal threshold depends on which embedder built the index.
        if (
            isinstance(llm, MockLLMClient)
            and retriever is not None
            and retriever.metadata["embedder"].startswith("hash-")
        ):
            llm.score_threshold = settings.mock_score_threshold_hash
        app.state.retriever = retriever
        app.state.llm = llm
        yield

    app = FastAPI(title="Policy Helpdesk", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        """Liveness/readiness: 503 until an index is loaded, so Docker won't route early."""
        retriever: Retriever | None = request.app.state.retriever
        body = {
            "status": "ok" if retriever else "degraded",
            "index_loaded": retriever is not None,
            "chunks": retriever.index.ntotal if retriever else 0,
            "embedder": retriever.metadata["embedder"] if retriever else None,
            "index_built_at": retriever.metadata["built_at"] if retriever else None,
            "llm_provider": request.app.state.llm.name,
        }
        return JSONResponse(body, status_code=200 if retriever else 503)

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        """Answer a question from the policy documents."""
        started = time.perf_counter()
        retriever: Retriever | None = request.app.state.retriever
        if retriever is None:
            raise HTTPException(503, "index not built; run: python -m ingest.build_index")

        chunks = retriever.retrieve(body.question)
        try:
            result = request.app.state.llm.generate(body.question, chunks)
        except LLMError as exc:
            logger.error("llm failure: %s", exc)
            raise HTTPException(502, "language model request failed") from exc

        # Sources are always returned (even for "I don't know") so scores can be inspected.
        sources = [Source(doc=c.doc, chunk_id=c.chunk_id, score=round(c.score, 4)) for c in chunks]
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return AskResponse(answer=result.answer, sources=sources, latency_ms=latency_ms)

    return app
