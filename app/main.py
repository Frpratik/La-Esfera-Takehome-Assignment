"""FastAPI application: /ask, /feedback, /stats, /health.

Run with: uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.cost import estimate_cost_usd
from app.db import init_db, make_engine, make_session_factory
from app.llm import LLMError, LLMResult, MockLLMClient, get_llm_client
from app.models import Feedback, QueryLog
from app.retriever import RetrievedChunk, Retriever
from app.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    Source,
    StatsResponse,
)
from app.stats import compute_stats

logger = logging.getLogger("helpdesk")


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests whose X-API-Key header is missing or wrong (401)."""
    expected = request.app.state.settings.api_key
    # compare_digest avoids leaking the key length/prefix through response timing.
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def get_db(request: Request) -> Iterator[Session]:
    """Yield a DB session for one request and always close it."""
    with request.app.state.session_factory() as session:
        yield session


def _log_query(
    request: Request,
    question: str,
    result: LLMResult,
    chunks: list[RetrievedChunk],
    sources: list[Source],
    latency_ms: float,
) -> int | None:
    """Persist the query; return its id, or None if the DB is unavailable.

    Logging must not take the service down: the user still gets their answer.
    """
    app_state = request.app.state
    provider = app_state.llm.name
    row = QueryLog(
        question=question,
        answer=result.answer,
        sources=[s.model_dump() for s in sources],
        top_score=chunks[0].score if chunks else None,
        latency_ms=latency_ms,
        provider=provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        est_cost_usd=estimate_cost_usd(
            app_state.settings, provider, result.prompt_tokens, result.completion_tokens
        ),
    )
    try:
        with app_state.session_factory() as session:
            session.add(row)
            session.commit()
            return row.id
    except SQLAlchemyError:
        logger.exception("failed to log query")
        return None


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
        init_db(app.state.engine)
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
    app.state.engine = make_engine(settings.database_url)
    app.state.session_factory = make_session_factory(app.state.engine)

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
        query_id = _log_query(request, body.question, result, chunks, sources, latency_ms)
        return AskResponse(
            query_id=query_id, answer=result.answer, sources=sources, latency_ms=latency_ms
        )

    @app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_api_key)])
    def feedback(body: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackResponse:
        """Store a thumbs up/down for an answer; re-voting replaces the earlier rating."""
        if db.get(QueryLog, body.query_id) is None:
            raise HTTPException(404, f"unknown query_id {body.query_id}")
        existing = db.query(Feedback).filter(Feedback.query_id == body.query_id).one_or_none()
        if existing:
            existing.rating = body.rating
        else:
            db.add(Feedback(query_id=body.query_id, rating=body.rating))
        db.commit()
        return FeedbackResponse(query_id=body.query_id, rating=body.rating)

    @app.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
    def stats(db: Session = Depends(get_db)) -> StatsResponse:
        """Usage, latency percentiles, estimated cost and thumbs-up rate."""
        return compute_stats(db)

    return app
