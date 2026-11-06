"""Tests for the evaluation set and the eval scoring logic."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from eval.run_eval import (
    facts_present,
    format_table,
    is_refusal,
    load_questions,
    rescore,
    run_eval,
    score_case,
    summarize,
)

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


# --- the question set itself -------------------------------------------------------------


def test_question_set_shape() -> None:
    questions = load_questions()
    assert len(questions) == 12
    assert sum(q["answerable"] for q in questions) == 9
    assert len({q["id"] for q in questions}) == 12


REGRESSION = Path(__file__).resolve().parent.parent / "eval" / "regression_questions.json"


@pytest.mark.parametrize("path", [None, REGRESSION], ids=["main", "regression"])
def test_expected_docs_and_facts_exist_in_the_policies(path: Path | None) -> None:
    """Guards against the ground truth drifting away from the documents."""
    for q in load_questions(path) if path else load_questions():
        if not q["answerable"]:
            assert q["expected_doc"] is None and q["expected_facts"] == []
            continue
        text = (POLICIES / f"{q['expected_doc']}.md").read_text(encoding="utf-8").lower()
        assert q["expected_facts"], q["id"]
        for fact in q["expected_facts"]:
            assert fact.lower() in text, f"{q['id']}: {fact!r} not in {q['expected_doc']}"


def test_answerable_questions_cover_all_eight_documents() -> None:
    docs = {q["expected_doc"] for q in load_questions() if q["answerable"]}
    assert docs == {p.stem for p in POLICIES.glob("*.md")}


# --- scoring ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I don't know", True),
        ("  i don't know.  ", True),
        ("I don’t know", True),
        ("I don't know the answer to that, sorry", False),
        ("The limit is $180", False),
    ],
)
def test_is_refusal(answer: str, expected: bool) -> None:
    assert is_refusal(answer) is expected


def test_facts_present_requires_all_facts_case_insensitively() -> None:
    assert facts_present("The IT MANAGER approves within 5 days", ["it manager", "5 days"])
    assert not facts_present("The IT Manager approves", ["IT Manager", "5 days"])


def question(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "t1",
        "question": "q?",
        "answerable": True,
        "expected_doc": "travel",
        "expected_facts": ["$180"],
    }
    return base | overrides


def response(answer: str, docs: list[str], latency: float = 5.0) -> dict[str, Any]:
    sources = [{"doc": d, "chunk_id": f"{d}-0", "score": 0.5} for d in docs]
    return {"answer": answer, "sources": sources, "latency_ms": latency}


def test_correct_answer_with_right_doc_passes() -> None:
    r = score_case(question(), response("Limit is $180 [travel-0]", ["travel", "expenses"]))
    assert (r.hit, r.top1, r.correct, r.refused, r.passed) == (True, True, True, False, True)


def test_right_doc_ranked_second_is_a_hit_but_not_top1() -> None:
    r = score_case(question(), response("Limit is $180", ["expenses", "travel"]))
    assert (r.hit, r.top1) == (True, False)


def test_right_doc_but_missing_fact_is_a_retrieval_hit_but_not_correct() -> None:
    r = score_case(question(), response("Hotels are limited.", ["travel"]))
    assert (r.hit, r.correct, r.passed) == (True, False, False)


def test_wrong_doc_is_a_retrieval_miss() -> None:
    r = score_case(question(), response("Limit is $180", ["expenses"]))
    assert r.hit is False


def test_refusing_an_answerable_question_is_not_correct() -> None:
    r = score_case(question(), response("I don't know", ["travel"]))
    assert (r.refused, r.correct, r.passed) == (True, False, False)


def test_unanswerable_passes_only_when_refused() -> None:
    q = question(answerable=False, expected_doc=None, expected_facts=[])
    assert score_case(q, response("I don't know", ["leave"])).passed is True
    wrong = score_case(q, response("Paris", ["leave"]))
    assert (wrong.passed, wrong.hit, wrong.top1, wrong.correct) == (False, None, None, None)


def test_summarize_computes_each_metric() -> None:
    answerable = question()
    unanswerable = question(answerable=False, expected_doc=None, expected_facts=[])
    results = [
        score_case(answerable, response("$180", ["travel"], 10)),  # hit + correct
        score_case(answerable, response("I don't know", ["expenses"], 20)),  # miss + refused
        score_case(unanswerable, response("I don't know", ["leave"], 30)),  # refused
        score_case(unanswerable, response("Paris", ["leave"], 40)),  # answered
    ]
    m = summarize(results, cost_per_query=0.002)
    assert m["retrieval_hit_rate"] == 0.5
    assert m["top1_doc_rate"] == 0.5
    assert m["answer_correctness"] == 0.5
    assert m["refusal_accuracy"] == 0.5
    assert m["false_refusal_rate"] == 0.5
    assert m["avg_latency_ms"] == 25.0 and m["max_latency_ms"] == 40.0
    assert m["est_cost_per_query_usd"] == 0.002
    assert "FAIL" in format_table(results, m)


def test_summarize_handles_empty_groups() -> None:
    m = summarize([], 0.0)
    assert m["retrieval_hit_rate"] is None and m["avg_latency_ms"] is None


# --- end to end through the real app ------------------------------------------------------


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def test_run_eval_against_the_app(client: TestClient) -> None:
    questions = [
        question(question="What is the hotel limit per night?"),
        question(
            id="t2",
            question="zebra quantum xylophone",
            answerable=False,
            expected_doc=None,
            expected_facts=[],
        ),
    ]
    results, cost = run_eval(client, questions, "test-key")
    assert [r.passed for r in results] == [True, True]
    assert results[0].retrieved_docs[0] == "travel"
    assert cost == 0.0  # mock provider is free


# --- scoring robustness (found in the live Groq run) ------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "File it in **TravelDesk** within **21 days**.",
        "File it in TravelDesk within 21\u202fdays.",  # narrow no-break space
        "File it in TravelDesk within 21\u00a0days.",  # no-break space
        "File it in `TravelDesk`\nwithin 21   days.",
    ],
)
def test_facts_match_through_markdown_and_unicode_spaces(answer: str) -> None:
    assert facts_present(answer, ["TravelDesk", "21 days"])


def test_rescore_reapplies_scoring_without_calling_the_api() -> None:
    q = question(expected_facts=["21 days"])
    saved = {
        "metrics": {"est_cost_per_query_usd": 0.0001},
        "cases": [
            {
                "id": "t1",
                "answer": "within 21\u202fdays",
                "retrieved_docs": ["travel"],
                "latency_ms": 1000.0,
                # stale verdicts from an older scorer:
                "passed": False,
                "correct": False,
            }
        ],
    }
    report = rescore(saved, [q])
    assert report["cases"][0]["passed"] is True
    assert report["metrics"]["answer_correctness"] == 1.0
    assert report["metrics"]["est_cost_per_query_usd"] == 0.0001
