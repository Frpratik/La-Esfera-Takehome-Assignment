"""Persist and load the FAISS index plus its JSON metadata."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ingest.chunker import STRATEGY, Chunk

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


def _atomic_write(path: Path, writer: Any) -> None:
    """Write via a temp file then rename, so a crash never leaves a half-written index."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def write_index(
    index_dir: Path,
    vectors: np.ndarray,
    chunks: list[Chunk],
    doc_hashes: dict[str, str],
    embedder_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    """Build a FAISS index from ``vectors`` and save it with metadata. Returns the metadata."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Inner product over unit vectors == cosine similarity; flat index is exact and fast
    # enough for a few dozen chunks.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    metadata: dict[str, Any] = {
        "built_at": datetime.now(UTC).isoformat(),
        "embedder": embedder_name,
        "dim": int(vectors.shape[1]),
        "chunking_strategy": STRATEGY,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # doc name -> content hash, used later to detect a stale index
        "documents": doc_hashes,
        # position i in this list == vector i in the FAISS index
        "chunks": [{"doc": c.doc, "chunk_id": c.chunk_id, "text": c.text} for c in chunks],
    }
    _atomic_write(index_dir / INDEX_FILE, lambda p: faiss.write_index(index, str(p)))
    _atomic_write(
        index_dir / METADATA_FILE,
        lambda p: p.write_text(json.dumps(metadata, indent=2), encoding="utf-8"),
    )
    return metadata


def load_index(index_dir: Path) -> tuple[faiss.Index, dict[str, Any]]:
    """Load the FAISS index and metadata saved by :func:`write_index`."""
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / INDEX_FILE))
    metadata = json.loads((index_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return index, metadata
