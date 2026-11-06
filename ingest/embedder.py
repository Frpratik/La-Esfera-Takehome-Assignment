"""Text embedders: sentence-transformers, with a deterministic hashing fallback."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Tiny stopword list: without IDF weighting, words like "the" would dominate the hash vectors.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...


class HashEmbedder:
    """Deterministic bag-of-words embedder using the hashing trick.

    No model download and no randomness, so it works offline and gives stable test results.
    Quality is lower than a trained model (no semantics, only shared words).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # A sign bit from the hash lets colliding tokens cancel instead of always adding up.
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]
            # Bigrams add a little word-order signal ("business class" vs "class business").
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for feature in features:
                idx, sign = self._bucket(feature)
                vectors[row, idx] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Cosine similarity == inner product only if vectors are unit length.
        return vectors / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        # The method was renamed in newer sentence-transformers; support both to avoid a
        # deprecation warning without breaking older installs.
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def embedder_from_name(name: str) -> Embedder:
    """Recreate the embedder recorded in index metadata.

    Queries must be embedded with the same model that built the index, otherwise the
    vectors live in different spaces and scores are meaningless.
    """
    if name.startswith("hash-"):
        return HashEmbedder(dim=int(name.split("-", 1)[1]))
    return SentenceTransformerEmbedder(name)


def get_embedder(model_name: str, use_hash: bool) -> Embedder:
    """Return the configured embedder, falling back to hashing if the model can't load."""
    if use_hash:
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # network failure, missing package, bad model name...
        logger.warning("could not load %s (%s); using hash embedder", model_name, exc)
        return HashEmbedder()
