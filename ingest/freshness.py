"""Detect whether the search index is out of date with the policy documents.

A stale index silently serves old facts after a policy edit, so the service compares what the
index recorded at build time (per-document hashes, chunking settings) with what is on disk now.

Usage: python -m ingest.freshness      # exit code 1 if the index is stale
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from ingest.chunker import STRATEGY
from ingest.documents import load_documents
from ingest.index_store import load_index


@dataclass(frozen=True)
class Freshness:
    """Differences between an index and the documents it should reflect."""

    changed: list[str] = field(default_factory=list)  # content hash differs
    added: list[str] = field(default_factory=list)  # on disk, not in the index
    removed: list[str] = field(default_factory=list)  # in the index, gone from disk
    settings_changed: list[str] = field(default_factory=list)  # chunking differs from config

    @property
    def stale(self) -> bool:
        return bool(self.changed or self.added or self.removed or self.settings_changed)

    @property
    def stale_documents(self) -> list[str]:
        return sorted({*self.changed, *self.added, *self.removed})

    def describe(self) -> str:
        """One-line human summary, used in logs and the CLI."""
        if not self.stale:
            return "index is up to date"
        parts = [
            f"{label}: {', '.join(names)}"
            for label, names in (
                ("changed", self.changed),
                ("added", self.added),
                ("removed", self.removed),
                ("settings", self.settings_changed),
            )
            if names
        ]
        return "index is STALE (" + "; ".join(parts) + ")"


def check_freshness(
    metadata: dict[str, Any], policies_dir: Path, chunk_size: int, chunk_overlap: int
) -> Freshness:
    """Compare index ``metadata`` with the documents in ``policies_dir`` and chunk settings."""
    indexed: dict[str, str] = metadata.get("documents", {})
    current = {d.name: d.sha256 for d in load_documents(policies_dir)}

    # Chunking changes (new strategy or different size/overlap) make every chunk outdated even
    # when no document changed, so they count as staleness too.
    settings_changed = []
    if metadata.get("chunking_strategy") != STRATEGY:
        settings_changed.append(
            f"chunking strategy {metadata.get('chunking_strategy')!r} -> {STRATEGY!r}"
        )
    if (metadata.get("chunk_size"), metadata.get("chunk_overlap")) != (chunk_size, chunk_overlap):
        settings_changed.append(
            f"chunk size/overlap {metadata.get('chunk_size')}/{metadata.get('chunk_overlap')}"
            f" -> {chunk_size}/{chunk_overlap}"
        )

    return Freshness(
        changed=sorted(n for n in current if n in indexed and current[n] != indexed[n]),
        added=sorted(n for n in current if n not in indexed),
        removed=sorted(n for n in indexed if n not in current),
        settings_changed=settings_changed,
    )


def main() -> int:
    """CLI: print freshness of the configured index; exit 1 if stale."""
    settings = get_settings()
    try:
        _, metadata = load_index(settings.index_dir)
    except FileNotFoundError:
        print(f"no index in {settings.index_dir}; run: python -m ingest.build_index")
        return 1
    result = check_freshness(
        metadata, settings.policies_dir, settings.chunk_size, settings.chunk_overlap
    )
    print(f"built_at={metadata['built_at']}  {result.describe()}")
    if result.stale:
        print("fix: python -m ingest.build_index")
    return 1 if result.stale else 0


if __name__ == "__main__":
    sys.exit(main())
