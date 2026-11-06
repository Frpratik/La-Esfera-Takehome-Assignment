"""Validate that policy markdown files meet the word-count requirements.

Usage: python -m ingest.validate_policies [--dir data/policies]
Exits with status 1 if any file is outside the allowed range or the count is wrong.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MIN_WORDS = 200
MAX_WORDS = 400
EXPECTED_DOCS = 8
DEFAULT_DIR = Path("data/policies")


def count_words(text: str) -> int:
    """Count words, ignoring markdown heading markers like '#' and '##'."""
    # Strip heading hashes so "## Title" counts as one word, not two tokens.
    cleaned = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return len(cleaned.split())


def validate(directory: Path) -> list[str]:
    """Return a list of human-readable problems (empty list means valid)."""
    files = sorted(directory.glob("*.md"))
    problems: list[str] = []
    if len(files) != EXPECTED_DOCS:
        problems.append(f"expected {EXPECTED_DOCS} policy files, found {len(files)}")
    for path in files:
        words = count_words(path.read_text(encoding="utf-8"))
        status = "ok" if MIN_WORDS <= words <= MAX_WORDS else "FAIL"
        print(f"{path.name:<24} {words:>4} words  {status}")
        if status == "FAIL":
            problems.append(f"{path.name}: {words} words (allowed {MIN_WORDS}-{MAX_WORDS})")
    return problems


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    problems = validate(args.dir)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
