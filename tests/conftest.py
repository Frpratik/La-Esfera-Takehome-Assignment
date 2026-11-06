"""Shared fixtures: a tiny policy corpus indexed with the offline hash embedder."""

from pathlib import Path

import pytest

from app.config import Settings
from ingest.build_index import build_index
from ingest.embedder import HashEmbedder

TRAVEL = (
    "# Travel\n"
    "The hotel limit is $180 per night. In New York the hotel limit is $250 per night. "
    "Travel claims must be submitted within 14 days of returning from the trip. "
    "Economy class is required for flights shorter than 6 hours."
)
LEAVE = (
    "# Leave\n"
    "Every employee receives 24 days of paid annual leave per year. "
    "Sick leave is 10 paid days and a doctor certificate is needed after 3 days."
)
EXPENSES = (
    "# Expenses\n"
    "Expense claims must be submitted within 30 days of purchase. "
    "Claims above $500 need approval from the Finance Director."
)


CHUNK_SIZE = 60
CHUNK_OVERLAP = 10


@pytest.fixture
def policies_dir(tmp_path: Path) -> Path:
    policies = tmp_path / "policies"
    policies.mkdir()
    for name, text in {"travel": TRAVEL, "leave": LEAVE, "expenses": EXPENSES}.items():
        (policies / f"{name}.md").write_text(text, encoding="utf-8")
    return policies


@pytest.fixture
def index_dir(tmp_path: Path, policies_dir: Path) -> Path:
    out = tmp_path / "index"
    build_index(policies_dir, out, CHUNK_SIZE, CHUNK_OVERLAP, HashEmbedder())
    return out


@pytest.fixture
def settings(index_dir: Path, policies_dir: Path, tmp_path: Path) -> Settings:
    # _env_file=None keeps a developer's local .env from leaking into tests.
    return Settings(
        _env_file=None,
        api_key="test-key",
        index_dir=index_dir,
        policies_dir=policies_dir,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        llm_provider="mock",
        mock_score_threshold_hash=0.10,
    )
