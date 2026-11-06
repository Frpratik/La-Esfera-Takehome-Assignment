"""Load policy documents from disk and fingerprint their content."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source policy file."""

    name: str  # file stem, e.g. "travel"
    text: str
    sha256: str


def content_hash(text: str) -> str:
    """SHA-256 of the text with line endings normalised.

    Normalising CRLF/LF keeps the hash identical on Windows checkouts and Linux containers,
    otherwise a git autocrlf difference would look like a content change (false stale alarm).
    """
    normalised = text.replace("\r\n", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def load_documents(policies_dir: Path) -> list[Document]:
    """Read every ``*.md`` file in ``policies_dir`` in a stable (sorted) order."""
    docs = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(name=path.stem, text=text, sha256=content_hash(text)))
    return docs
