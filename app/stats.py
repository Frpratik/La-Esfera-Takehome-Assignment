"""Aggregate statistics over logged queries and feedback."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Feedback, QueryLog
from app.schemas import StatsResponse


def percentile(values: list[float], pct: float) -> float | None:
    """Percentile with linear interpolation between ranks (same as numpy's default).

    Rank = pct/100 x (n-1); a fractional rank blends the two neighbouring values, so p50 of
    [1, 2, 3, 4] is 2.5 rather than jumping to 2 or 3. Tail percentiles (p95) matter more than
    the mean for latency because a few slow requests are hidden by averages.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = pct / 100 * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def compute_stats(session: Session) -> StatsResponse:
    """Build the /stats payload. Fields are None when there is no data to compute them."""
    count = session.scalar(select(func.count(QueryLog.id))) or 0
    total_cost = session.scalar(select(func.coalesce(func.sum(QueryLog.est_cost_usd), 0.0))) or 0.0
    # Percentiles are computed in Python because SQLite has no percentile function; fine at
    # this scale (for millions of rows use Postgres percentile_cont or a sampled window).
    latencies = list(session.scalars(select(QueryLog.latency_ms)))

    feedback_count = session.scalar(select(func.count(Feedback.id))) or 0
    thumbs_up = session.scalar(select(func.count(Feedback.id)).where(Feedback.rating == "up")) or 0

    def _round(value: float | None, digits: int) -> float | None:
        return None if value is None else round(value, digits)

    return StatsResponse(
        count=count,
        p50_latency_ms=_round(percentile(latencies, 50), 2),
        p95_latency_ms=_round(percentile(latencies, 95), 2),
        total_est_cost_usd=round(total_cost, 6),
        avg_est_cost_usd=round(total_cost / count, 6) if count else None,
        feedback_count=feedback_count,
        thumbs_up_rate=round(thumbs_up / feedback_count, 4) if feedback_count else None,
    )
