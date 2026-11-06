"""Checks that the synthetic policy corpus stays within the agreed size limits."""

from pathlib import Path

from ingest.validate_policies import count_words, validate

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def test_count_words_ignores_heading_markers() -> None:
    assert count_words("# Title\n\nsome body text") == 4


def test_policy_corpus_is_valid() -> None:
    assert validate(POLICIES) == []
