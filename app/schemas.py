"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated

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

    answer: str
    sources: list[Source]
    latency_ms: float
