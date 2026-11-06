"""A small BM25 lexical scorer (no dependencies, deterministic)."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9$]+")


def _normalise(token: str) -> str:
    # Drop a plural "s" so "claim"/"claims", "hotel"/"hotels" match. Without this, a question
    # in the singular scored ~0 against the passage that answered it in the plural, and rank
    # fusion pushed that passage (dense rank 1) out of the top-k. Words of 3 letters or fewer
    # ("is", "has") and "-ss" words ("business") are left alone.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, plural-normalised word tokens; '$' is kept so "$500" differs from "500"."""
    return [_normalise(t) for t in _TOKEN.findall(text.lower())]


class BM25:
    """Okapi BM25 over a fixed list of documents.

    Why it complements embeddings: a dense model blurs near-duplicate passages (two policies
    that both describe "reimbursement") into similar vectors, while BM25 rewards the exact,
    rare words that tell them apart ("TravelDesk", "ExpenseHub", "hotel"). IDF makes those
    rare terms count for more than words that appear in every document.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1  # how quickly repeated terms stop adding score
        self.b = b  # how strongly long documents are penalised
        self._docs = [Counter(tokenize(t)) for t in texts]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(texts)) if texts else 0.0
        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(doc.keys())
        n = len(texts)
        # The "+1" inside the log keeps IDF positive even for terms in most documents.
        self._idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for ``query`` (0.0 when no query term matches)."""
        terms = set(tokenize(query))
        out = []
        for freqs, length in zip(self._docs, self._lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[term] * tf * (self.k1 + 1) / norm
            out.append(score)
        return out
