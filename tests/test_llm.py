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
        OpenAICompatibleClient("http://llm.test", "", "m", sleep=lambda s: None).generate(
            "q", [CHUNK]
        )


def test_mock_finds_the_sentence_in_a_lower_ranked_chunk() -> None:
    first = RetrievedChunk("leave", "leave-0", "Employees receive 24 days of paid leave.", 0.8)
    second = RetrievedChunk(
        "leave",
        "leave-1",
        "Annual leave requests longer than 3 days need 10 working days of notice.",
        0.6,
    )
    answer = MockLLMClient(0.35).generate("How many days notice for annual leave?", [first, second])
    assert answer.answer == (
        "Annual leave requests longer than 3 days need 10 working days of notice. [leave-1]"
    )


def test_mock_prefers_higher_ranked_chunk_on_ties() -> None:
    a = RetrievedChunk("a", "a-0", "The hotel limit is $180 per night here.", 0.8)
    b = RetrievedChunk("b", "b-0", "The hotel limit is $250 per night there.", 0.7)
    answer = MockLLMClient(0.35).generate("hotel limit per night", [a, b]).answer
    assert answer.endswith("[a-0]")


def test_mock_ignores_chunks_beyond_top_n() -> None:
    chunks = [
        RetrievedChunk("x", f"x-{i}", "Unrelated filler sentence about nothing.", 0.9 - i / 10)
        for i in range(3)
    ]
    chunks.append(RetrievedChunk("y", "y-0", "Mileage is reimbursed at $0.58 per mile.", 0.5))
    assert MockLLMClient(0.35, top_n=3).generate("mileage per mile", chunks).answer == REFUSAL


def test_mock_matches_singular_question_to_plural_sentence() -> None:
    chunk = RetrievedChunk(
        "expenses",
        "expenses-1",
        "Northwind Facilities Expenses Policy. "
        "Approved expense claims are reimbursed by bank transfer within 10 business days.",
        0.8,
    )
    answer = MockLLMClient(0.35).generate("How long does an expense claim take?", [chunk]).answer
    assert "10 business days" in answer


# --- retries ------------------------------------------------------------------------------

OK_BODY = {"choices": [{"message": {"content": "24 days"}}], "usage": {}}


def scripted_post(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> list[str]:
    """Make httpx.post return ``responses`` in order; returns the list of called URLs."""
    calls: list[str] = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        calls.append(url)
        resp = responses[len(calls) - 1]
        resp.request = httpx.Request("POST", url)
        return resp

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def test_rate_limit_is_retried_using_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    calls = scripted_post(
        monkeypatch,
        [httpx.Response(429, headers={"retry-after": "2"}), httpx.Response(200, json=OK_BODY)],
    )
    client = OpenAICompatibleClient("http://llm.test", "k", "m", sleep=waits.append)
    assert client.generate("q", [CHUNK]).answer == "24 days"
    assert len(calls) == 2 and waits == [2.0]


def test_backoff_doubles_without_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    scripted_post(
        monkeypatch,
        [httpx.Response(503), httpx.Response(503), httpx.Response(200, json=OK_BODY)],
    )
    client = OpenAICompatibleClient("http://llm.test", "k", "m", backoff_s=0.5, sleep=waits.append)
    client.generate("q", [CHUNK])
    assert waits == [0.5, 1.0]


def test_retry_wait_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    scripted_post(
        monkeypatch,
        [httpx.Response(429, headers={"retry-after": "600"}), httpx.Response(200, json=OK_BODY)],
    )
    OpenAICompatibleClient("http://llm.test", "k", "m", sleep=waits.append).generate("q", [CHUNK])
    assert waits == [OpenAICompatibleClient.MAX_RETRY_WAIT_S]


def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = scripted_post(monkeypatch, [httpx.Response(429, text="slow down")] * 3)
    client = OpenAICompatibleClient(
        "http://llm.test", "k", "m", max_retries=2, sleep=lambda s: None
    )
    with pytest.raises(LLMError, match="429"):
        client.generate("q", [CHUNK])
    assert len(calls) == 3


def test_client_errors_are_not_retried_and_keep_the_provider_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"error": {"message": "The model does not exist", "code": "model_not_found"}}
    calls = scripted_post(monkeypatch, [httpx.Response(404, json=body)])
    client = OpenAICompatibleClient("http://llm.test", "k", "gone", sleep=lambda s: None)
    with pytest.raises(LLMError, match="model_not_found"):
        client.generate("q", [CHUNK])
    assert len(calls) == 1


def test_malformed_response_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted_post(monkeypatch, [httpx.Response(200, json={"choices": []})])
    with pytest.raises(LLMError, match="unexpected response"):
        OpenAICompatibleClient("http://llm.test", "k", "m").generate("q", [CHUNK])
