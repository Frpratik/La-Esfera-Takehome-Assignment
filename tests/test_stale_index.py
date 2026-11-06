"""Regression tests for the stale-index incident (see docs/DEBUGGING.md).

A policy was edited but the index was not rebuilt, so the service kept answering with the old
text. These tests fail if the service ever serves an index older than the documents again.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from ingest.freshness import check_freshness
from ingest.index_store import load_index
from tests.conftest import CHUNK_OVERLAP, CHUNK_SIZE

HEADERS = {"X-API-Key": "test-key"}
QUESTION = {"question": "What is the hotel limit per night?"}
REPO = Path(__file__).resolve().parent.parent


def edit_hotel_limit(policies_dir: Path) -> None:
    """Simulate a policy update: the hotel limit goes from $180 to $200."""
    path = policies_dir / "travel.md"
    path.write_text(path.read_text(encoding="utf-8").replace("$180", "$200"), encoding="utf-8")


def freshness(index_dir: Path, policies_dir: Path, size: int = CHUNK_SIZE):
    _, metadata = load_index(index_dir)
    return check_freshness(metadata, policies_dir, size, CHUNK_OVERLAP)


# --- detection ----------------------------------------------------------------------------


def test_untouched_index_is_fresh(index_dir: Path, policies_dir: Path) -> None:
    assert not freshness(index_dir, policies_dir).stale


def test_edited_document_is_reported_as_changed(index_dir: Path, policies_dir: Path) -> None:
    edit_hotel_limit(policies_dir)
    result = freshness(index_dir, policies_dir)
    assert result.stale and result.changed == ["travel"]
    assert "travel" in result.describe()


def test_added_and_removed_documents_are_reported(index_dir: Path, policies_dir: Path) -> None:
    (policies_dir / "security.md").write_text("# Security\nUse MFA.", encoding="utf-8")
    (policies_dir / "leave.md").unlink()
    result = freshness(index_dir, policies_dir)
    assert result.added == ["security"] and result.removed == ["leave"]


def test_line_ending_change_alone_is_not_stale(index_dir: Path, policies_dir: Path) -> None:
    path = policies_dir / "travel.md"
    # Normalise first: on Windows the file may already use CRLF.
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert not freshness(index_dir, policies_dir).stale


def test_changed_chunk_settings_make_the_index_stale(index_dir: Path, policies_dir: Path) -> None:
    result = freshness(index_dir, policies_dir, size=CHUNK_SIZE + 1)
    assert result.stale and result.settings_changed


def test_index_from_another_chunking_strategy_is_stale(policies_dir: Path, index_dir: Path) -> None:
    _, metadata = load_index(index_dir)
    old = metadata | {"chunking_strategy": "words-v0"}
    assert check_freshness(old, policies_dir, CHUNK_SIZE, CHUNK_OVERLAP).stale
    missing = {k: v for k, v in metadata.items() if k != "chunking_strategy"}
    assert check_freshness(missing, policies_dir, CHUNK_SIZE, CHUNK_OVERLAP).stale


# --- the service --------------------------------------------------------------------------


def test_health_reports_a_document_edited_while_running(
    settings: Settings, policies_dir: Path
) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "ok"
        edit_hotel_limit(policies_dir)
        resp = client.get("/health")
    body = resp.json()
    assert resp.status_code == 200  # still serving, but flagged
    assert (body["status"], body["index_stale"], body["stale_documents"]) == (
        "stale",
        True,
        ["travel"],
    )


def test_stale_index_serves_the_old_fact_when_auto_rebuild_is_off(
    settings: Settings, policies_dir: Path
) -> None:
    """Documents the failure: without a rebuild the answer still says $180."""
    edit_hotel_limit(policies_dir)
    off = settings.model_copy(update={"auto_rebuild_index": False})
    with TestClient(create_app(off)) as client:
        answer = client.post("/ask", json=QUESTION, headers=HEADERS).json()["answer"]
        health = client.get("/health").json()
    assert "$180" in answer and "$200" not in answer
    assert health["index_stale"] is True


def test_startup_rebuilds_a_stale_index_so_answers_use_the_new_text(
    settings: Settings, policies_dir: Path, index_dir: Path
) -> None:
    """Fails on the pre-fix service, which kept answering from the old index."""
    built_before = load_index(index_dir)[1]["built_at"]
    edit_hotel_limit(policies_dir)
    with TestClient(create_app(settings)) as client:
        answer = client.post("/ask", json=QUESTION, headers=HEADERS).json()["answer"]
        health = client.get("/health").json()
    assert "$200" in answer and "$180" not in answer
    assert health["status"] == "ok" and health["index_stale"] is False
    assert load_index(index_dir)[1]["built_at"] != built_before


# --- the real local index -----------------------------------------------------------------


def test_local_index_is_not_stale_against_the_policies() -> None:
    """Fails if data/policies changed since data/index was built. Skipped when no index exists."""
    settings = Settings(_env_file=None)
    index_path = REPO / settings.index_dir
    if not (index_path / "metadata.json").exists():
        pytest.skip("no local index built")
    _, metadata = load_index(index_path)
    result = check_freshness(
        metadata, REPO / settings.policies_dir, settings.chunk_size, settings.chunk_overlap
    )
    assert not result.stale, f"{result.describe()} - run: python -m ingest.build_index"
