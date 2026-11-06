"""Retrieval over the FAISS index: dense similarity, optionally fused with BM25."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from app.bm25 import BM25
from ingest.embedder import Embedder, embedder_from_name
from ingest.index_store import load_index

# Standard RRF constant: damps the advantage of rank 1 over rank 2 so that agreement between
# two rankings counts for more than a single very high rank in one of them.
RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned for a query.

    ``score`` is always the dense cosine similarity (used for refusal thresholds and shown in
    the UI). In hybrid mode the *order* comes from rank fusion, so scores need not decrease.
    """

    doc: str
    chunk_id: str
    text: str
    score: float
    bm25: float = 0.0


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge several best-first rankings of item ids into one.

    Each item scores sum(1 / (k + rank)) over the rankings it appears in. Using ranks, not raw
    scores, means cosine similarities and BM25 values never need to be put on one scale.
    Ties are broken by the earlier ranking, so the result is deterministic.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    # Tie-break on the earliest position an item holds across the rankings (dense first).
    first_seen: dict[int, int] = {}
    position = 0
    for ranking in rankings:
        for item in ranking:
            first_seen.setdefault(item, position)
            position += 1
    return sorted(fused, key=lambda item: (-fused[item], first_seen[item]))


class Retriever:
    """Embeds a question and returns the top-k most relevant chunks."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: dict[str, Any],
        embedder: Embedder,
        top_k: int,
        mode: str = "dense",
    ) -> None:
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"unknown retrieval mode: {mode!r}")
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.top_k = top_k
        self.mode = mode
        self._bm25 = BM25([c["text"] for c in metadata["chunks"]])

    @classmethod
    def from_dir(cls, index_dir: Path, top_k: int, mode: str = "dense") -> Retriever:
        """Load the saved index and the embedder it was built with."""
        index, metadata = load_index(index_dir)
        return cls(index, metadata, embedder_from_name(metadata["embedder"]), top_k, mode)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, best first."""
        total = self.index.ntotal
        # Rank *every* chunk (the corpus is tiny) so fusion sees full rankings.
        sims, ids = self.index.search(self.embedder.embed([question]), total)
        dense_scores = {int(i): float(s) for s, i in zip(sims[0], ids[0], strict=True) if i >= 0}
        dense_order = [int(i) for i in ids[0] if i >= 0]

        bm25_scores = self._bm25.scores(question)
        if self.mode == "hybrid":
            # Only chunks sharing at least one term take part in the lexical ranking;
            # ranking zero-score chunks would just add noise.
            lexical = sorted(
                (i for i, s in enumerate(bm25_scores) if s > 0),
                key=lambda i: (-bm25_scores[i], i),
            )
            order = reciprocal_rank_fusion([dense_order, lexical])
        else:
            order = dense_order

        results = []
        for idx in order[: self.top_k]:
            chunk = self.metadata["chunks"][idx]
            results.append(
                RetrievedChunk(
                    chunk["doc"],
                    chunk["chunk_id"],
                    chunk["text"],
                    dense_scores[idx],
                    bm25_scores[idx],
                )
            )
        return results
