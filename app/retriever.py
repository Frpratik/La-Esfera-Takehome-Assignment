"""Vector retrieval over the FAISS index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from ingest.embedder import Embedder, embedder_from_name
from ingest.index_store import load_index


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned for a query, with its cosine similarity score."""

    doc: str
    chunk_id: str
    text: str
    score: float


class Retriever:
    """Embeds a question and returns the top-k most similar chunks."""

    def __init__(
        self, index: faiss.Index, metadata: dict[str, Any], embedder: Embedder, top_k: int
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.top_k = top_k

    @classmethod
    def from_dir(cls, index_dir: Path, top_k: int) -> Retriever:
        """Load the saved index and the embedder it was built with."""
        index, metadata = load_index(index_dir)
        return cls(index, metadata, embedder_from_name(metadata["embedder"]), top_k)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, best first."""
        vector = self.embedder.embed([question])
        scores, ids = self.index.search(vector, min(self.top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0:  # FAISS pads with -1 when fewer results exist
                continue
            chunk = self.metadata["chunks"][int(idx)]
            results.append(
                RetrievedChunk(chunk["doc"], chunk["chunk_id"], chunk["text"], float(score))
            )
        return results
