"""Prompt template for the policy assistant."""

from __future__ import annotations

from app.retriever import RetrievedChunk

# The exact refusal string; the mock LLM, API and eval all compare against it.
REFUSAL = "I don't know"

# Design: keep the rules short and unambiguous. Small models follow a strict output
# contract ("reply exactly ...") better than a vague "be careful" instruction.
SYSTEM_PROMPT = f"""You are a company policy assistant.
Answer the question using ONLY the context below. Do not use outside knowledge.
Cite the sources you used in square brackets, like [travel-1].
If the context does not contain the answer, reply exactly: {REFUSAL}"""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as labelled context followed by the question."""
    context = "\n\n".join(f"[{c.chunk_id}] ({c.doc})\n{c.text}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
