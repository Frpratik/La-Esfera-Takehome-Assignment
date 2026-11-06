"""Tests for BM25, rank fusion and hybrid retrieval."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from app.bm25 import BM25, tokenize
from app.retriever import Retriever, reciprocal_rank_fusion
from ingest.chunker import Chunk
from ingest.index_store import load_index, write_index

# --- BM25 ---------------------------------------------------------------------------------


def test_tokenize_keeps_dollar_amounts_distinct() -> None:
    assert tokenize("Limit is $500, not 500!") == ["limit", "is", "$500", "not", "500"]


def test_bm25_ranks_the_document_with_the_rare_term_first() -> None:
    bm25 = BM25(
        [
            "claims are reimbursed by bank transfer",
            "claims must be submitted in traveldesk",
            "claims must be submitted in expensehub",
        ]
    )
    scores = bm25.scores("how do I submit traveldesk claims")
    assert scores.index(max(scores)) == 1


def test_bm25_gives_zero_when_no_term_matches() -> None:
    assert BM25(["alpha beta", "gamma delta"]).scores("zebra") == [0.0, 0.0]


def test_bm25_penalises_longer_documents_with_the_same_term_count() -> None:
    short = "hotel limit"
    long = "hotel limit " + "filler " * 50
    scores = BM25([short, long]).scores("hotel")
    assert scores[0] > scores[1]


def test_bm25_handles_an_empty_corpus() -> None:
    assert BM25([]).scores("anything") == []


# --- rank fusion --------------------------------------------------------------------------


def test_fusion_promotes_items_ranked_well_by_both_lists() -> None:
    # item 2 is 2nd in both lists (2/62); items 0 and 3 are 1st in only one (1/61 each).
    assert reciprocal_rank_fusion([[0, 2, 1], [3, 2]])[0] == 2


def test_two_rankings_that_swap_a_pair_tie_and_the_first_list_wins() -> None:
    # Documents a property of equal-weight RRF: A>B in one list and B>A in the other is a tie.
    assert reciprocal_rank_fusion([[0, 1], [1, 0]]) == [0, 1]


def test_fusion_includes_items_from_either_list() -> None:
    assert set(reciprocal_rank_fusion([[0, 1], [2]])) == {0, 1, 2}


def test_fusion_is_deterministic_on_ties() -> None:
    # 0 and 1 each rank first in one list: tie -> the earlier list's item comes first.
    assert reciprocal_rank_fusion([[0], [1]]) == [0, 1]
    assert reciprocal_rank_fusion([[1], [0]]) == [1, 0]


# --- hybrid vs dense retrieval ------------------------------------------------------------


class StubEmbedder:
    """Returns fixed vectors so a test controls the dense ranking exactly."""

    name = "stub"
    dim = 3

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._table[t] for t in texts], dtype=np.float32)


@pytest.fixture
def near_duplicate_index(tmp_path: Path) -> Callable[[str], Retriever]:
    """Factory building a retriever (given a mode) over two near-duplicate chunks.

    Dense search prefers the expenses chunk, but only the travel chunk shares the query's
    exact terms (e.g. "TravelDesk"), so BM25 does not rank the expenses chunk at all.
    """
    chunks = [
        Chunk("expenses", "expenses-0", "Payments arrive by bank wire shortly after sign-off."),
        Chunk("travel", "travel-0", "Approved claims are reimbursed via TravelDesk transfer."),
        Chunk("leave", "leave-0", "Employees receive twenty four days of annual leave."),
    ]
    vectors = np.array([[1, 0, 0], [0.9, 0.436, 0], [0, 0, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    write_index(tmp_path, vectors, chunks, {}, "stub", 120, 30)
    index, metadata = load_index(tmp_path)
    embedder = StubEmbedder({"how are travel claims reimbursed in traveldesk": [1, 0, 0]})
    return lambda mode: Retriever(index, metadata, embedder, top_k=3, mode=mode)


def test_dense_mode_ranks_by_similarity_only(near_duplicate_index) -> None:
    hits = near_duplicate_index("dense").retrieve("how are travel claims reimbursed in traveldesk")
    assert [h.chunk_id for h in hits][:2] == ["expenses-0", "travel-0"]


def test_hybrid_mode_lets_the_exact_term_break_the_near_duplicate_tie(
    near_duplicate_index,
) -> None:
    hits = near_duplicate_index("hybrid").retrieve("how are travel claims reimbursed in traveldesk")
    assert hits[0].chunk_id == "travel-0"
    assert hits[0].bm25 > 0


def test_scores_stay_dense_cosine_in_hybrid_mode(near_duplicate_index) -> None:
    hits = near_duplicate_index("hybrid").retrieve("how are travel claims reimbursed in traveldesk")
    by_id = {h.chunk_id: h.score for h in hits}
    assert by_id["expenses-0"] == pytest.approx(1.0, abs=1e-5)  # unchanged by fusion


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    chunks = [Chunk("a", "a-0", "alpha")]
    write_index(tmp_path, np.array([[1, 0, 0]], dtype=np.float32), chunks, {}, "stub", 120, 30)
    index, metadata = load_index(tmp_path)
    with pytest.raises(ValueError):
        Retriever(index, metadata, StubEmbedder({}), 1, mode="nope")
