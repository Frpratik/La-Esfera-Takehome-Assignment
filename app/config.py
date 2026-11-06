"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    api_key: str = "change-me"
    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    # LLM
    llm_provider: str = "mock"  # mock | openai
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    mock_score_threshold: float = 0.35  # tuned for MiniLM (answerable >= 0.5, unanswerable <= 0.25)
    mock_score_threshold_hash: float = 0.10  # hash-embedder scores are on a much lower scale
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_hash_embedder: bool = False

    # Index / retrieval
    policies_dir: Path = Path("data/policies")
    index_dir: Path = Path("data/index")
    chunk_size: int = 120
    chunk_overlap: int = 30
    top_k: int = 4
    # dense = embeddings only; hybrid = embeddings + BM25 fused with reciprocal rank fusion.
    retrieval_mode: str = "hybrid"
    # Rebuild the index at startup if documents changed since it was built (else just warn).
    auto_rebuild_index: bool = True

    # Database
    database_url: str = "sqlite:///./helpdesk.db"

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list (the env var is comma separated)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
