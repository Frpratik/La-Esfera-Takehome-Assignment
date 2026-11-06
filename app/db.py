"""Database engine and session factory (SQLite for dev/tests, Postgres in Docker)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base


def make_engine(database_url: str) -> Engine:
    """Create an engine; SQLite needs check_same_thread off because FastAPI uses a thread pool."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    # pool_pre_ping revalidates connections so a restarted Postgres doesn't break the API.
    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create tables if missing (no migrations: schema is small and owned by this service)."""
    Base.metadata.create_all(engine)
