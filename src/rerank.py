from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from src.utils import env_flag


STOPWORDS = {
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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
}


@dataclass
class RerankInput:
    """Minimal reranker input shared with retrieval."""

    text: str
    score: float


def tokenize_query(text: str) -> list[str]:
    """Tokenize text for heuristic reranking."""

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def heuristic_rerank_score(query: str, text: str, embedding_score: float) -> float:
    """Blend embedding score with lightweight lexical and phrase evidence."""

    query_terms = tokenize_query(query)
    text_terms = set(tokenize_query(text))
    if not query_terms:
        return embedding_score

    overlap = sum(1 for term in query_terms if term in text_terms)
    overlap_score = overlap / len(query_terms)

    query_phrase = " ".join(query_terms)
    normalized_text = " ".join(tokenize_query(text))
    phrase_boost = 0.12 if query_phrase and query_phrase in normalized_text else 0.0

    density_boost = min(0.1, len(query_terms) / max(40, len(normalized_text.split())))
    return (0.72 * embedding_score) + (0.28 * overlap_score) + phrase_boost + density_boost


@lru_cache(maxsize=1)
def load_cross_encoder(model_name: str):
    """Lazily load an optional cross-encoder reranker."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def can_use_cross_encoder(model_name: str | None) -> bool:
    """Return whether cross-encoder reranking is configured and allowed."""

    return bool(model_name) and env_flag("RAG_ENABLE_RERANK", default=True)


def rerank_scores(
    query: str,
    items: Iterable[RerankInput],
    model_name: str | None = None,
) -> list[float]:
    """Return rerank scores, using a cross-encoder when configured."""

    candidates = list(items)
    if not candidates:
        return []

    if can_use_cross_encoder(model_name):
        try:
            cross_encoder = load_cross_encoder(model_name)
            pairs = [[query, item.text] for item in candidates]
            raw_scores = cross_encoder.predict(pairs)
            return [float(1 / (1 + math.exp(-value))) for value in raw_scores]
        except Exception:
            pass

    return [
        heuristic_rerank_score(query=query, text=item.text, embedding_score=item.score)
        for item in candidates
    ]
