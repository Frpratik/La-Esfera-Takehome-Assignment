"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    """Body of POST /ask."""

    question: Question


class Source(BaseModel):
    """A retrieved chunk cited alongside an answer."""

    doc: str
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    """Body returned by POST /ask."""

    # None only if the DB write failed: the answer is still returned, but can't get feedback.
    query_id: int | None
    answer: str
    sources: list[Source]
    latency_ms: float
    # True when the answer came from the cache (no retrieval or LLM call was made).
    cached: bool = False


class FeedbackRequest(BaseModel):
    """Body of POST /feedback."""

    query_id: int
    rating: Literal["up", "down"]


class FeedbackResponse(BaseModel):
    """Acknowledgement of a stored rating."""

    query_id: int
    rating: Literal["up", "down"]


class StatsResponse(BaseModel):
    """Body returned by GET /stats. Optional fields are None when there is no data."""

    count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    total_est_cost_usd: float
    avg_est_cost_usd: float | None
    feedback_count: int
    thumbs_up_rate: float | None
