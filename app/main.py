"""FastAPI application: /ask, /feedback, /stats, /health, /metrics.

Run with: uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.cost import estimate_cost_usd
from app.db import init_db, make_engine, make_session_factory
from app.llm import LLMError, LLMResult, MockLLMClient, get_llm_client
from app.metrics import Metrics
from app.models import Feedback, QueryLog
from app.prompt import REFUSAL
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
from ingest.build_index import build_index
from ingest.freshness import check_freshness

logger = logging.getLogger("helpdesk")


def _configure_logging() -> None:
    """Send helpdesk INFO logs to stderr once (uvicorn only configures its own loggers)."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


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
    cost_usd: float,
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
        top_score=max((c.score for c in chunks), default=None),
        latency_ms=latency_ms,
        provider=provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        est_cost_usd=cost_usd,
    )
    try:
        with app_state.session_factory() as session:
            session.add(row)
            session.commit()
            return row.id
    except SQLAlchemyError:
        logger.exception("failed to log query")
        return None


def _is_refusal(answer: str) -> bool:
    """True for the refusal string, tolerating case and a trailing period from real LLMs."""
    return answer.strip().replace("’", "'").rstrip(".").lower() == REFUSAL.lower()


def _route_label(request: Request) -> str:
    """The matched route template ("/ask"), never the raw URL.

    Raw paths would let anyone create unlimited metric series by requesting random URLs, so
    anything that matched no route is grouped under "unmatched".
    """
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def _load_retriever(settings: Settings) -> Retriever | None:
    """Load the index, or return None (service starts degraded) if it isn't available."""
    try:
        return Retriever.from_dir(settings.index_dir, settings.top_k, settings.retrieval_mode)
    except Exception as exc:  # missing/corrupt index, embedder failure...
        logger.warning("index not loaded from %s: %s", settings.index_dir, exc)
        return None


def _ensure_fresh(retriever: Retriever, settings: Settings) -> Retriever:
    """Warn if the index is stale and, when enabled, rebuild it and return a fresh retriever.

    Serving an index built from older documents silently returns outdated facts, so the
    mismatch is always logged; AUTO_REBUILD_INDEX decides whether to repair it automatically.
    """
    freshness = check_freshness(
        retriever.metadata, settings.policies_dir, settings.chunk_size, settings.chunk_overlap
    )
    if not freshness.stale:
        return retriever
    logger.warning("%s", freshness.describe())
    if not settings.auto_rebuild_index:
        logger.warning("AUTO_REBUILD_INDEX is off: serving the stale index")
        return retriever
    try:
        # Reuse the embedder the old index used so scores stay comparable to its thresholds.
        build_index(
            settings.policies_dir,
            settings.index_dir,
            settings.chunk_size,
            settings.chunk_overlap,
            retriever.embedder,
        )
    except Exception:
        logger.exception("automatic index rebuild failed: serving the stale index")
        return retriever
    logger.warning("index rebuilt from %s", settings.policies_dir)
    return _load_retriever(settings) or retriever


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Takes explicit settings so tests can inject their own."""
    settings = settings or get_settings()
    _configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(app.state.engine)
        retriever = _load_retriever(settings)
        if retriever is not None:
            retriever = _ensure_fresh(retriever, settings)
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

    app = FastAPI(title="Northwind Facilities Helpdesk", lifespan=lifespan)
    app.state.settings = settings
    # The browser UI runs on another origin, so it needs CORS. Origins are an explicit
    # allow-list from env (never "*"), because requests carry an API key header.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    app.state.engine = make_engine(settings.database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.metrics = metrics = Metrics()

    @app.middleware("http")
    async def record_request_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Count and time every request, labelled by route template and status."""
        started = time.perf_counter()
        status = 500  # if the handler raises, the client receives a 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = _route_label(request)
            metrics.http_requests.labels(request.method, route, str(status)).inc()
            metrics.http_latency.labels(request.method, route).observe(
                time.perf_counter() - started
            )

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        """Liveness/readiness: 503 until an index is loaded, so Docker won't route early.

        A stale index still answers (HTTP 200) but reports status "stale" and names the
        documents that changed, so monitoring can alert without taking the service down.
        """
        retriever: Retriever | None = request.app.state.retriever
        settings: Settings = request.app.state.settings
        freshness = (
            check_freshness(
                retriever.metadata,
                settings.policies_dir,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            if retriever
            else None
        )
        status = "degraded" if retriever is None else "stale" if freshness.stale else "ok"
        body = {
            "status": status,
            "index_loaded": retriever is not None,
            "index_stale": freshness.stale if freshness else None,
            "stale_documents": freshness.stale_documents if freshness else [],
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
            metrics.answers.labels("llm_error").inc()
            raise HTTPException(502, "language model request failed") from exc
        provider = request.app.state.llm.name
        cost_usd = estimate_cost_usd(
            request.app.state.settings, provider, result.prompt_tokens, result.completion_tokens
        )
        metrics.answers.labels("refused" if _is_refusal(result.answer) else "answered").inc()
        metrics.llm_tokens.labels("prompt").inc(result.prompt_tokens)
        metrics.llm_tokens.labels("completion").inc(result.completion_tokens)
        metrics.llm_cost.inc(cost_usd)

        # Sources are always returned (even for "I don't know") so scores can be inspected.
        sources = [Source(doc=c.doc, chunk_id=c.chunk_id, score=round(c.score, 4)) for c in chunks]
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        # One line per request with the retrieval evidence (chunk ids + scores), so a bad
        # answer can be traced to retrieval without re-running the question.
        logger.info(
            "ask q=%r top=[%s] answer=%r",
            body.question[:80],
            ", ".join(f"{c.chunk_id}:{c.score:.2f}" for c in chunks),
            result.answer[:80],
        )
        query_id = _log_query(request, body.question, result, chunks, sources, latency_ms, cost_usd)
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

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics(request: Request) -> Response:
        """Prometheus scrape endpoint (unauthenticated, like /health: keep it on the internal
        network). Index gauges are refreshed at scrape time so a policy edited while the
        service runs shows up without waiting for a request."""
        retriever: Retriever | None = request.app.state.retriever
        metrics.index_loaded.set(1 if retriever else 0)
        if retriever is not None:
            fresh = check_freshness(
                retriever.metadata,
                settings.policies_dir,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            metrics.index_stale.set(1 if fresh.stale else 0)
        payload, content_type = metrics.render()
        return Response(content=payload, media_type=content_type)

    return app
