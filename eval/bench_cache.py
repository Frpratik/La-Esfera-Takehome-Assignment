"""Measure the answer cache: latency of a first (uncached) vs repeated (cached) question.

Asks every eval question twice through /ask against a real Redis. The first pass misses the
cache (retrieval + LLM), the second hits it. Uses a throwaway database and FLUSHES the given
Redis database first, so point it at a Redis you do not mind clearing.

Usage:
    docker run -d --rm -p 6390:6379 --name bench-redis redis:7-alpine
    python -m eval.bench_cache --redis redis://localhost:6390/0
    LLM_PROVIDER=openai python -m eval.bench_cache --redis redis://localhost:6390/0
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import redis
from fastapi.testclient import TestClient

from app.cache import RedisCache
from app.config import get_settings
from app.main import create_app
from eval.run_eval import QUESTIONS_PATH, load_questions


def _summary(values: list[float]) -> str:
    return f"p50 {statistics.median(values):8.1f} ms   mean {statistics.fmean(values):8.1f} ms"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--redis", required=True, help="Redis URL to use (it will be FLUSHED)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    client = redis.Redis.from_url(args.redis)
    client.flushdb()
    settings = get_settings()
    questions = [q["question"] for q in load_questions(args.questions)]
    headers = {"X-API-Key": settings.api_key}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_url = f"sqlite:///{Path(tmp, 'bench.db').as_posix()}"
        app = create_app(settings.model_copy(update={"database_url": db_url}))
        app.state.cache = RedisCache(args.redis, settings.cache_ttl_seconds, client=client)
        with TestClient(app) as api:
            health = api.get("/health").json()
            api.post("/ask", json={"question": "warm-up question"}, headers=headers)
            results: dict[str, dict[str, list[float]]] = {}
            for label in ("miss", "hit"):
                server, wall = [], []
                for question in questions:
                    started = time.perf_counter()
                    body = api.post("/ask", json={"question": question}, headers=headers).json()
                    wall.append((time.perf_counter() - started) * 1000)
                    server.append(body["latency_ms"])
                    expected_cached = label == "hit"
                    if body["cached"] is not expected_cached:
                        print(f"unexpected cached={body['cached']} for {question!r}")
                results[label] = {"server": server, "wall": wall}
        app.state.engine.dispose()

    print(f"LLM: {health['llm_provider']}   embedder: {health['embedder']}")
    print(f"questions: {len(questions)}")
    for label, title in (("miss", "first ask (cache miss)"), ("hit", "repeat ask (cache hit)")):
        print(f"{title:<24} server {_summary(results[label]['server'])}")
        print(f"{'':<24} wall   {_summary(results[label]['wall'])}")
    speedup = statistics.median(results["miss"]["wall"]) / statistics.median(results["hit"]["wall"])
    print(f"median end-to-end speed-up on repeats: {speedup:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
