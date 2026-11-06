"""Build the FAISS index from the policy documents.

Usage: python -m ingest.build_index [--policies-dir DIR] [--index-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from ingest.chunker import Chunk, chunk_document
from ingest.documents import load_documents
from ingest.embedder import Embedder, get_embedder
from ingest.index_store import write_index

logger = logging.getLogger(__name__)


def build_index(
    policies_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder,
) -> dict[str, Any]:
    """Chunk, embed and index all documents; return the written metadata."""
    docs = load_documents(policies_dir)
    if not docs:
        raise FileNotFoundError(f"no .md documents found in {policies_dir}")

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc.name, doc.text, chunk_size, chunk_overlap))

    vectors = embedder.embed([c.text for c in chunks])
    return write_index(
        index_dir,
        vectors,
        chunks,
        {d.name: d.sha256 for d in docs},
        embedder.name,
        chunk_size,
        chunk_overlap,
    )


def main() -> int:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies-dir", type=Path, default=settings.policies_dir)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    embedder = get_embedder(settings.embedding_model, settings.use_hash_embedder)
    try:
        meta = build_index(
            args.policies_dir,
            args.index_dir,
            settings.chunk_size,
            settings.chunk_overlap,
            embedder,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"indexed {len(meta['documents'])} documents, {len(meta['chunks'])} chunks "
        f"with {meta['embedder']} at {meta['built_at']} -> {args.index_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
