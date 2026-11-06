"""Cost estimation from token counts."""

from __future__ import annotations

from app.config import Settings


def estimate_cost_usd(
    settings: Settings, provider: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimated USD cost = tokens / 1000 x per-1K price, for prompt and completion separately.

    The mock provider is always free, even if prices are configured.
    """
    if provider == "mock":
        return 0.0
    return (
        prompt_tokens / 1000 * settings.price_per_1k_prompt
        + completion_tokens / 1000 * settings.price_per_1k_completion
    )
