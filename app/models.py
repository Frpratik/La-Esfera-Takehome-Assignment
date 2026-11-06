"""SQLAlchemy models for query logging and user feedback."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    # Python-side default keeps behaviour identical on SQLite and Postgres.
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all tables."""


class QueryLog(Base):
    """One row per /ask call."""

    __tablename__ = "queries"
    # Index choice: created_at is the column we filter and order by for time-windowed
    # stats and "most recent queries" views. Without it those scans grow with the table.
    __table_args__ = (Index("ix_queries_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(32))
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    est_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Feedback(Base):
    """A thumbs up/down on one answer (at most one row per query)."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN ('up', 'down')", name="ck_feedback_rating"),
        # Index choice: Postgres does not index foreign keys automatically. This one serves
        # the join/lookup from a query to its feedback and cascade deletes. It is UNIQUE
        # because we keep one rating per query (re-voting updates the row).
        Index("ix_feedback_query_id", "query_id", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    rating: Mapped[str] = mapped_column(String(4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
