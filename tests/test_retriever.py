"""Tests for the retriever."""

from pathlib import Path

from app.retriever import Retriever


def test_top_result_is_the_relevant_document(index_dir: Path) -> None:
    retriever = Retriever.from_dir(index_dir, top_k=3)
    hits = retriever.retrieve("What is the hotel limit per night?")
    assert hits[0].doc == "travel"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_top_k_is_capped_by_index_size(index_dir: Path) -> None:
    retriever = Retriever.from_dir(index_dir, top_k=50)
    assert len(retriever.retrieve("annual leave")) == retriever.index.ntotal
