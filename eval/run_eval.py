"""Evaluate the helpdesk on a fixed question set.

Reports retrieval hit rate, answer correctness, refusal accuracy, latency and estimated cost,
prints a table and writes eval/results.json.

Usage (needs a built index: python -m ingest.build_index):
    python -m eval.run_eval                       # in-process, offline, throwaway database
    python -m eval.run_eval --url http://localhost:8000 --api-key change-me   # running service
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.prompt import REFUSAL

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


@dataclass
class CaseResult:
    """Outcome for one eval question."""

    id: str
    question: str
    answerable: bool
    expected_doc: str | None
    retrieved_docs: list[str]
    hit: bool | None  # expected doc in the retrieved sources (answerable only)
    correct: bool | None  # answered, with every expected fact present (answerable only)
    refused: bool
    passed: bool  # answerable: correct; unanswerable: refused
    answer: str
    latency_ms: float


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Read the eval question file."""
    return json.loads(path.read_text(encoding="utf-8"))


def is_refusal(answer: str) -> bool:
    """True if the answer is the refusal string (tolerating case, trailing period, curly quote)."""
    normalised = answer.strip().replace("’", "'").rstrip(".").lower()
    return normalised == REFUSAL.lower()


def facts_present(answer: str, facts: list[str]) -> bool:
    """Simple fact match: every expected fact appears in the answer (case-insensitive)."""
    lowered = answer.lower()
    return all(fact.lower() in lowered for fact in facts)


def score_case(question: dict[str, Any], response: dict[str, Any]) -> CaseResult:
    """Score one /ask response against its expected doc and facts."""
    answer: str = response["answer"]
    docs = [s["doc"] for s in response["sources"]]
    refused = is_refusal(answer)
    answerable: bool = question["answerable"]

    if answerable:
        hit: bool | None = question["expected_doc"] in docs
        # A refusal never counts as correct, even if it happened to contain a fact string.
        correct: bool | None = (not refused) and facts_present(answer, question["expected_facts"])
        passed = bool(correct)
    else:
        hit, correct = None, None
        passed = refused

    return CaseResult(
        id=question["id"],
        question=question["question"],
        answerable=answerable,
        expected_doc=question["expected_doc"],
        retrieved_docs=docs,
        hit=hit,
        correct=correct,
        refused=refused,
        passed=passed,
        answer=answer,
        latency_ms=response["latency_ms"],
    )


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def summarize(results: list[CaseResult], cost_per_query: float) -> dict[str, Any]:
    """Aggregate per-case results into the headline metrics."""
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    latencies = [r.latency_ms for r in results]
    return {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        # Did retrieval put the right document in the top-k? Independent of the LLM.
        "retrieval_hit_rate": _rate([bool(r.hit) for r in answerable]),
        # Did the final answer contain the expected facts?
        "answer_correctness": _rate([bool(r.correct) for r in answerable]),
        # Did the system refuse the questions it should not be able to answer?
        "refusal_accuracy": _rate([r.refused for r in unanswerable]),
        # Wrongly refusing a question that has an answer (the cost of being too cautious).
        "false_refusal_rate": _rate([r.refused for r in answerable]),
        "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else None,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
        "est_cost_per_query_usd": round(cost_per_query, 6),
    }


def run_eval(
    client: httpx.Client, questions: list[dict[str, Any]], api_key: str
) -> tuple[list[CaseResult], float]:
    """Ask every question through the API. Returns case results and the cost per query."""
    headers = {"X-API-Key": api_key}

    def total_cost() -> float:
        resp = client.get("/stats", headers=headers)
        resp.raise_for_status()
        return float(resp.json()["total_est_cost_usd"])

    # /ask does not return cost, so measure the change in the service's running total.
    cost_before = total_cost()
    results = []
    for q in questions:
        resp = client.post("/ask", json={"question": q["question"]}, headers=headers)
        resp.raise_for_status()
        results.append(score_case(q, resp.json()))
    cost_per_query = (total_cost() - cost_before) / len(questions) if questions else 0.0
    return results, cost_per_query


def format_table(results: list[CaseResult], metrics: dict[str, Any]) -> str:
    """Render per-question rows plus a summary block as plain text."""
    lines = [f"{'ID':<4} {'RESULT':<6} {'DOC HIT':<8} {'LATENCY':>9}  QUESTION"]
    for r in results:
        hit = "-" if r.hit is None else ("yes" if r.hit else "NO")
        lines.append(
            f"{r.id:<4} {'PASS' if r.passed else 'FAIL':<6} {hit:<8} "
            f"{r.latency_ms:>7.1f}ms  {r.question[:60]}"
        )
    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "Failures:"]
        for r in failures:
            lines.append(f"  {r.id} expected_doc={r.expected_doc} retrieved={r.retrieved_docs}")
            lines.append(f"      answer: {r.answer}")

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines += [
        "",
        f"Retrieval hit rate : {pct(metrics['retrieval_hit_rate'])}",
        f"Answer correctness : {pct(metrics['answer_correctness'])}",
        f"Refusal accuracy   : {pct(metrics['refusal_accuracy'])}",
        f"False refusal rate : {pct(metrics['false_refusal_rate'])}",
        f"Avg latency        : {metrics['avg_latency_ms']} ms "
        f"(p50 {metrics['p50_latency_ms']}, max {metrics['max_latency_ms']})",
        f"Est. cost / query  : ${metrics['est_cost_per_query_usd']:.6f}",
    ]
    return "\n".join(lines)


def _evaluate(
    client: httpx.Client, api_key: str, questions: list[dict[str, Any]]
) -> dict[str, Any]:
    health = client.get("/health").json()
    if not health.get("index_loaded"):
        raise SystemExit("index not loaded; run: python -m ingest.build_index")
    results, cost = run_eval(client, questions, api_key)
    metrics = summarize(results, cost)
    print(format_table(results, metrics))
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": health["embedder"],
        "llm_provider": health["llm_provider"],
        "top_k": max((len(r.retrieved_docs) for r in results), default=0),
        "metrics": metrics,
        "cases": [asdict(r) for r in results],
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", help="base URL of a running API (default: run in-process)")
    parser.add_argument("--api-key", help="API key (default: API_KEY from the environment/.env)")
    parser.add_argument("--index-dir", type=Path, help="index directory for in-process mode")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    settings = get_settings()
    api_key = args.api_key or settings.api_key
    questions = load_questions(args.questions)

    if args.url:
        with httpx.Client(base_url=args.url, timeout=60) as client:
            report = _evaluate(client, api_key, questions)
    else:
        # In-process: no server needed, and a temp database keeps eval runs out of real logs.
        from fastapi.testclient import TestClient

        from app.main import create_app

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            update: dict[str, Any] = {
                "database_url": f"sqlite:///{Path(tmp, 'eval.db').as_posix()}"
            }
            if args.index_dir:
                update["index_dir"] = args.index_dir
            app = create_app(settings.model_copy(update=update))
            with TestClient(app) as client:
                report = _evaluate(client, api_key, questions)
            app.state.engine.dispose()  # release the SQLite file so the temp dir can be removed

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
