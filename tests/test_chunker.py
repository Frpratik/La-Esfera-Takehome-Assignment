"""Unit tests for the chunker."""

import pytest

from ingest.chunker import chunk_document, chunk_text


def numbered(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("  \n ", 10, 2) == []


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text(numbered(5), 10, 2) == [numbered(5)]


def test_consecutive_chunks_share_overlap_words() -> None:
    chunks = chunk_text(numbered(100), 40, 10)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.split()[-10:] == nxt.split()[:10]


def test_all_words_covered_and_no_tail_only_chunk() -> None:
    chunks = chunk_text(numbered(100), 40, 10)
    assert chunks[-1].split()[-1] == "w99"
    # step is 30: starts 0, 30, 60 -> the window at 60 reaches word 99, so no 4th chunk
    assert len(chunks) == 3


def test_exact_fit_does_not_add_extra_chunk() -> None:
    assert len(chunk_text(numbered(40), 40, 10)) == 1


def test_heading_markers_are_removed_and_headings_end_with_period() -> None:
    assert chunk_text("# Title\n## Section\nbody", 10, 0) == ["Title. Section. body"]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size, overlap)


def test_chunk_document_assigns_ids() -> None:
    chunks = chunk_document("travel", numbered(100), 40, 10)
    assert [c.chunk_id for c in chunks] == ["travel-0", "travel-1", "travel-2"]
    assert {c.doc for c in chunks} == {"travel"}
