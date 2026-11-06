"""Tests for the prompt and the LLM clients."""

import httpx
import pytest

from app.config import Settings
from app.llm import (
    LLMError,
    MockLLMClient,
    OpenAICompatibleClient,
    get_llm_client,
)
from app.prompt import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from app.retriever import RetrievedChunk

CHUNK = RetrievedChunk(
    doc="expenses",
    chunk_id="expenses-0",
    text=(
        "Claims are paid in 10 days. Mileage is reimbursed at $0.58 per mile. Fines are never paid."
    ),
    score=0.8,
)


def test_prompt_contains_rules_context_and_question() -> None:
    prompt = build_user_prompt("How much per mile?", [CHUNK])
    assert REFUSAL in SYSTEM_PROMPT
    assert "[expenses-0]" in prompt and "How much per mile?" in prompt


def test_mock_returns_best_matching_sentence_with_citation() -> None:
    result = MockLLMClient(0.35).generate("How much is mileage per mile?", [CHUNK])
    assert result.answer == "Mileage is reimbursed at $0.58 per mile. [expenses-0]"
    assert result.prompt_tokens > 0 and result.completion_tokens > 0


def test_mock_skips_heading_fragments() -> None:
    chunk = RetrievedChunk(
        "it", "it-0", "Refresh cycle. Laptops are replaced every 36 months.", 0.9
    )
    answer = MockLLMClient(0.35).generate("What is the laptop refresh cycle?", [chunk]).answer
    assert answer == "Laptops are replaced every 36 months. [it-0]"


def test_mock_refuses_below_threshold() -> None:
    weak = RetrievedChunk("expenses", "expenses-0", CHUNK.text, 0.2)
    assert MockLLMClient(0.35).generate("mileage?", [weak]).answer == REFUSAL


def test_mock_refuses_when_no_sentence_matches() -> None:
    assert MockLLMClient(0.35).generate("zebra quantum", [CHUNK]).answer == REFUSAL


def test_mock_refuses_with_no_chunks() -> None:
    assert MockLLMClient(0.35).generate("anything", []).answer == REFUSAL


def test_provider_selection_and_unknown_provider() -> None:
    assert get_llm_client(Settings(_env_file=None, llm_provider="mock")).name == "mock"
    assert get_llm_client(Settings(_env_file=None, llm_provider="openai")).name == "openai"
    with pytest.raises(ValueError):
        get_llm_client(Settings(_env_file=None, llm_provider="nope"))


def test_openai_client_parses_response_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        seen.update(url=url, **kwargs)
        body = {
            "choices": [{"message": {"content": " 24 days [leave-0] "}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient("http://llm.test/v1/", "secret", "m")
    result = client.generate("leave?", [CHUNK])

    assert (result.answer, result.prompt_tokens, result.completion_tokens) == (
        "24 days [leave-0]",
        100,
        5,
    )
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    assert seen["json"]["temperature"] == 0


def test_openai_client_wraps_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMError):
        OpenAICompatibleClient("http://llm.test", "", "m").generate("q", [CHUNK])
