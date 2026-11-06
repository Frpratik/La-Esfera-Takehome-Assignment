"""Tests for hashing, embedding and index building (hash embedder, fully offline)."""

from pathlib import Path

import numpy as np

from ingest.build_index import build_index
from ingest.documents import content_hash, load_documents
from ingest.embedder import HashEmbedder
from ingest.index_store import load_index


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    emb = HashEmbedder()
    a = emb.embed(["laptop refresh every 36 months"])
    b = emb.embed(["laptop refresh every 36 months"])
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_hash_embedder_ranks_overlapping_text_higher() -> None:
    emb = HashEmbedder()
    texts = ["hotel limit per night", "hotel limit is 180 per night", "sick leave"]
    q, close, far = emb.embed(texts)
    assert float(q @ close) > float(q @ far)


def test_content_hash_ignores_line_endings() -> None:
    assert content_hash("a\r\nb\n") == content_hash("a\nb\n")
    assert content_hash("a") != content_hash("b")


def test_build_index_writes_chunks_hashes_and_timestamp(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "one.md").write_text("# One\n" + "alpha beta gamma " * 40, encoding="utf-8")
    (policies / "two.md").write_text("# Two\nshort doc about hotels", encoding="utf-8")

    meta = build_index(policies, tmp_path / "idx", 50, 10, HashEmbedder())
    index, loaded = load_index(tmp_path / "idx")

    assert index.ntotal == len(loaded["chunks"]) == len(meta["chunks"])
    assert set(loaded["documents"]) == {"one", "two"}
    docs = {d.name: d.sha256 for d in load_documents(policies)}
    assert loaded["documents"] == docs
    assert loaded["built_at"] and loaded["embedder"] == "hash-384"
