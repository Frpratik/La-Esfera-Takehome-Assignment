"""LLM clients: a deterministic offline mock and an OpenAI-compatible HTTP client."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings
from app.prompt import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from app.retriever import RetrievedChunk


class LLMError(RuntimeError):
    """Raised when the upstream LLM call fails."""


@dataclass(frozen=True)
class LLMResult:
    """An answer plus token counts used for cost estimation."""

    answer: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    """Interface every provider implements."""

    name: str

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        """Answer ``question`` from ``chunks`` (or reply with the refusal string)."""
        ...


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for providers that omit usage."""
    return max(1, len(text) // 4)


_WORD = re.compile(r"[a-z0-9$]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "much",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)


def _stem(word: str) -> str:
    # Crude stemmer for a deterministic offline mock: drop a plural "s" so "claims" matches
    # "claim" (truncating alone made them differ), then cut to 6 chars so "approves" matches
    # "approval" and "expenses" matches "expense".
    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]
    return word[:6]


def _keywords(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def _split_sentences(text: str) -> list[str]:
    # Split on ., ! or ? followed by whitespace; requiring whitespace keeps decimals
    # like 0.58 intact.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class MockLLMClient:
    """Deterministic, offline stand-in: returns the best-matching sentence from the top chunks."""

    name = "mock"

    def __init__(self, score_threshold: float, top_n: int = 3) -> None:
        self.score_threshold = score_threshold
        # How many of the best-ranked chunks to search for the answer sentence.
        self.top_n = top_n

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        prompt_tokens = estimate_tokens(SYSTEM_PROMPT + build_user_prompt(question, chunks))
        answer = self._answer(question, chunks)
        return LLMResult(answer, prompt_tokens, estimate_tokens(answer))

    def _answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        # Retrieval score gates the answer: a weak top match means the corpus likely
        # doesn't cover the question, so refuse rather than quote an unrelated sentence.
        if not chunks or max(c.score for c in chunks) < self.score_threshold:
            return REFUSAL
        wanted = _keywords(question)
        best, best_chunk, best_overlap = "", chunks[0], 0
        # The top chunk is not always the one holding the fact (a leave chunk may rank first
        # while the notice rule sits in the next one), so search the top few chunks.
        for chunk in chunks[: self.top_n]:
            for sentence in _split_sentences(chunk.text):
                if len(sentence.split()) < 4:  # headings ("Refresh cycle.") are not answers
                    continue
                overlap = len(wanted & _keywords(sentence))
                # strict '>' keeps the earlier (higher-ranked) chunk and sentence on ties
                if overlap > best_overlap:
                    best, best_chunk, best_overlap = sentence, chunk, overlap
        if not best:
            return REFUSAL
        return f"{best} [{best_chunk.chunk_id}]"


class OpenAICompatibleClient:
    """Chat-completions client for OpenAI-compatible servers (Groq, Ollama, OpenAI...)."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        user_prompt = build_user_prompt(question, chunks)
        payload = {
            "model": self.model,
            "temperature": 0,  # factual QA: we want repeatable answers
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            answer = body["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        usage = body.get("usage") or {}
        return LLMResult(
            answer,
            usage.get("prompt_tokens") or estimate_tokens(SYSTEM_PROMPT + user_prompt),
            usage.get("completion_tokens") or estimate_tokens(answer),
        )


def get_llm_client(settings: Settings) -> LLMClient:
    """Select the provider from ``LLM_PROVIDER`` (mock | openai)."""
    if settings.llm_provider == "mock":
        return MockLLMClient(settings.mock_score_threshold)
    if settings.llm_provider == "openai":
        return OpenAICompatibleClient(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model
        )
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider!r}")
