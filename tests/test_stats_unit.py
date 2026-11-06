"""Unit tests for the percentile helper and cost estimation."""

import pytest

from app.config import Settings
from app.cost import estimate_cost_usd
from app.stats import percentile


def test_percentile_of_empty_list_is_none() -> None:
    assert percentile([], 50) is None


def test_percentile_single_value() -> None:
    assert percentile([7.0], 95) == 7.0


def test_percentile_interpolates_between_ranks() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    # rank = 0.95 * 3 = 2.85 -> 3 + 0.85 * (4 - 3)
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)


def test_percentile_ignores_input_order() -> None:
    assert percentile([4, 1, 3, 2], 50) == 2.5


def test_cost_is_tokens_over_1000_times_price() -> None:
    s = Settings(_env_file=None, price_per_1k_prompt=0.5, price_per_1k_completion=1.5)
    assert estimate_cost_usd(s, "openai", 2000, 1000) == pytest.approx(2.5)


def test_mock_provider_is_always_free() -> None:
    s = Settings(_env_file=None, price_per_1k_prompt=0.5, price_per_1k_completion=1.5)
    assert estimate_cost_usd(s, "mock", 5000, 5000) == 0.0
