"""Debugging aid: show what retrieval returns for a question, with scores.

Usage:
    python -m app.debug "How do I get reimbursed for a business trip?" [--k 6] [--full]

Prints the top-k chunks (score, id, document, text) and the answer the mock LLM would give,
so a wrong answer can be traced to retrieval (wrong chunks / scores) or to generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.llm import MockLLMClient
from app.retriever import Retriever


def main() -> int:
    """CLI entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=6, help="how many chunks to show")
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--full", action="store_true", help="print whole chunk text")
    args = parser.parse_args()

    try:
        retriever = Retriever.from_dir(args.index_dir, args.k)
    except FileNotFoundError:
        print(f"no index in {args.index_dir}; run: python -m ingest.build_index", file=sys.stderr)
        return 1

    meta = retriever.metadata
    print(
        f"index   : built_at={meta['built_at']} embedder={meta['embedder']} "
        f"chunks={retriever.index.ntotal}"
    )
    print(f"question: {args.question}\n")
    hits = retriever.retrieve(args.question)
    print(f"{'#':>2} {'score':>6}  {'chunk_id':<22} text")
    for rank, hit in enumerate(hits, start=1):
        text = hit.text if args.full else hit.text.replace("\n", " ")[:100] + "..."
        print(f"{rank:>2} {hit.score:>6.3f}  {hit.chunk_id:<22} {text}")

    # Use the same refusal threshold the API would pick for this embedder.
    hashed = meta["embedder"].startswith("hash-")
    threshold = settings.mock_score_threshold_hash if hashed else settings.mock_score_threshold
    answer = MockLLMClient(threshold).generate(args.question, hits).answer
    print(f"\nmock answer (threshold {threshold}): {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
