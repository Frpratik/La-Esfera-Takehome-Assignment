"""Split policy documents into overlapping word-window chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a source document."""

    doc: str  # document name (file stem, e.g. "travel")
    chunk_id: str  # unique id, e.g. "travel-2"
    text: str


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of ``size`` words, each overlapping the previous by ``overlap``.

    Markdown heading markers are dropped so '##' is not embedded as a word.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    # Drop '#' markers and end each heading with a period: otherwise headings glue onto
    # the next sentence once newlines are flattened ("Annual leave Every employee ...").
    words = re.sub(r"^#+\s*(.+?)\s*$", r"\1.", text, flags=re.MULTILINE).split()
    # Step is size - overlap: each window re-reads `overlap` words from the previous one,
    # so a sentence cut by a boundary still appears whole in at least one chunk
    # (as long as it is shorter than the overlap).
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        # Stop once a window reaches the end; otherwise the loop would emit a final
        # chunk made only of words already covered by the overlap.
        if start + size >= len(words):
            break
    return chunks


def chunk_document(doc: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Chunk one document and assign stable ids of the form ``<doc>-<index>``."""
    return [
        Chunk(doc=doc, chunk_id=f"{doc}-{i}", text=piece)
        for i, piece in enumerate(chunk_text(text, size, overlap))
    ]
