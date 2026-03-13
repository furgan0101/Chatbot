from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from src.embed import generate_embeddings, load_embedding_model, load_index
from src.ingest import load_documents
from src.rerank import RerankInput, rerank_scores
from src.utils import INDEX_FILE, METADATA_FILE, env_flag, env_float, env_int, load_json


DEFAULT_INITIAL_RETRIEVAL_K = 10
DEFAULT_FINAL_RETRIEVAL_K = 4
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_SCORE_THRESHOLD = 0.2


@dataclass
class RetrievedChunk:
    """A retrieved chunk with similarity score and source metadata."""

    chunk_id: str
    source_name: str
    source_path: str
    text: str
    start_char: int
    end_char: int
    score: float
    retrieval_score: float
    page_number: int | None = None
    section_heading: str | None = None
    source_variants: list[str] | None = None


@dataclass
class CorpusState:
    """State of the indexed/uploaded corpus used by the chatbot and UI."""

    documents_count: int
    chunks_indexed: int
    has_documents: bool
    has_index: bool
    has_indexed_corpus: bool


@lru_cache(maxsize=2)
def _load_metadata_cached(metadata_path: str, modified_time_ns: int) -> list[dict]:
    """Load metadata keyed by path and modification time to avoid repeated disk reads."""

    del modified_time_ns
    return load_json(Path(metadata_path))


def load_metadata() -> list[dict]:
    """Load persisted chunk metadata from disk."""

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_FILE}.")
    stat = METADATA_FILE.stat()
    return _load_metadata_cached(str(METADATA_FILE), stat.st_mtime_ns)


def get_corpus_state() -> CorpusState:
    """Return a shared corpus availability snapshot.

    This separates "no uploaded documents" from "documents exist but retrieval found no
    relevant chunks", so UI status and chatbot fallback wording rely on the same source
    of truth.
    """

    try:
        documents_count = len(load_documents())
    except Exception:
        documents_count = 0

    try:
        chunks_indexed = len(load_metadata()) if METADATA_FILE.exists() else 0
    except Exception:
        chunks_indexed = 0

    has_documents = documents_count > 0
    has_index = INDEX_FILE.exists()
    has_indexed_corpus = has_documents and has_index and chunks_indexed > 0
    return CorpusState(
        documents_count=documents_count,
        chunks_indexed=chunks_indexed,
        has_documents=has_documents,
        has_index=has_index,
        has_indexed_corpus=has_indexed_corpus,
    )


def normalize_chunk_text(text: str) -> str:
    """Normalize chunk text for duplicate filtering."""

    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def token_set(text: str) -> set[str]:
    """Return a simple token set for overlap checks."""

    return {token for token in normalize_chunk_text(text).split() if token}


def content_similarity(left: str, right: str) -> float:
    """Approximate near-duplicate similarity between two chunk texts."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def is_duplicate_candidate(candidate: RetrievedChunk, selected: list[RetrievedChunk]) -> bool:
    """Return True when a candidate is too similar to already selected chunks."""

    for existing in selected:
        if candidate.chunk_id == existing.chunk_id:
            return True
        if candidate.source_name == existing.source_name and candidate.page_number == existing.page_number:
            if content_similarity(candidate.text, existing.text) >= 0.82:
                return True
        if content_similarity(candidate.text, existing.text) >= 0.9:
            return True
    return False


def enforce_context_budget(
    chunks: list[RetrievedChunk],
    final_k: int,
    max_context_chars: int,
) -> list[RetrievedChunk]:
    """Keep as many strong unique chunks as fit in the configured context budget."""

    selected: list[RetrievedChunk] = []
    total_chars = 0
    for chunk in chunks:
        projected = total_chars + len(chunk.text)
        if selected and projected > max_context_chars and len(selected) >= min(3, final_k):
            continue
        selected.append(chunk)
        total_chars = projected
        if len(selected) >= final_k:
            break
    return selected


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    enable_rerank: bool | None = None,
    initial_k_override: int | None = None,
) -> List[RetrievedChunk]:
    """Retrieve, rerank, deduplicate, and trim chunks for a user question."""

    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    initial_k = initial_k_override or env_int("RAG_INITIAL_RETRIEVAL_K", DEFAULT_INITIAL_RETRIEVAL_K)
    final_k = top_k or env_int("RAG_FINAL_RETRIEVAL_K", DEFAULT_FINAL_RETRIEVAL_K)
    max_context_chars = env_int("RAG_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)
    min_score = score_threshold if score_threshold is not None else env_float(
        "RAG_RETRIEVAL_SCORE_THRESHOLD",
        DEFAULT_SCORE_THRESHOLD,
    )

    index = load_index()
    metadata = load_metadata()
    if index.ntotal == 0 or not metadata:
        return []

    model = load_embedding_model()
    query_embedding = generate_embeddings([cleaned_query], model)
    scores, indices = index.search(query_embedding, initial_k)

    candidates: list[RetrievedChunk] = []
    for retrieval_score, metadata_index in zip(scores[0], indices[0]):
        if metadata_index == -1 or metadata_index >= len(metadata):
            continue
        if float(retrieval_score) < min_score:
            continue

        chunk_data = metadata[metadata_index]
        candidates.append(
            RetrievedChunk(
                chunk_id=chunk_data["chunk_id"],
                source_name=chunk_data["source_name"],
                source_path=chunk_data["source_path"],
                text=chunk_data["text"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                score=float(retrieval_score),
                retrieval_score=float(retrieval_score),
                page_number=chunk_data.get("page_number"),
                section_heading=chunk_data.get("section_heading"),
                source_variants=chunk_data.get("source_variants", []),
            )
        )

    if not candidates:
        return []

    should_rerank = (
        enable_rerank if enable_rerank is not None else env_flag("RAG_ENABLE_RERANK", default=True)
    )

    if should_rerank:
        rerank_model = None
        if env_flag("RAG_USE_CROSS_ENCODER", default=False):
            rerank_model = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked_scores = rerank_scores(
            query=cleaned_query,
            items=[RerankInput(text=item.text, score=item.retrieval_score) for item in candidates],
            model_name=rerank_model,
        )
    else:
        reranked_scores = [item.retrieval_score for item in candidates]

    reranked_candidates = sorted(
        (
            RetrievedChunk(
                chunk_id=item.chunk_id,
                source_name=item.source_name,
                source_path=item.source_path,
                text=item.text,
                start_char=item.start_char,
                end_char=item.end_char,
                score=rerank_score,
                retrieval_score=item.retrieval_score,
                page_number=item.page_number,
                section_heading=item.section_heading,
                source_variants=item.source_variants,
            )
            for item, rerank_score in zip(candidates, reranked_scores)
        ),
        key=lambda item: (item.score, item.retrieval_score),
        reverse=True,
    )

    selected: list[RetrievedChunk] = []
    per_source_counts: dict[str, int] = {}
    for candidate in reranked_candidates:
        if is_duplicate_candidate(candidate, selected):
            continue

        source_count = per_source_counts.get(candidate.source_name, 0)
        if source_count >= 3 and len(selected) < len(reranked_candidates):
            continue

        selected.append(candidate)
        per_source_counts[candidate.source_name] = source_count + 1
        if len(selected) >= final_k * 2:
            break

    if not selected:
        return []

    selected = enforce_context_budget(
        chunks=selected,
        final_k=final_k,
        max_context_chars=max_context_chars,
    )

    return selected[:final_k]


def retrieve_corpus_overview_chunks(
    top_k: int | None = None,
    max_context_chars: int | None = None,
) -> List[RetrievedChunk]:
    """Return representative chunks across the indexed corpus for broad summary queries."""

    corpus_state = get_corpus_state()
    if not corpus_state.has_indexed_corpus:
        return []

    metadata = load_metadata()
    if not metadata:
        return []

    final_k = top_k or env_int("RAG_FINAL_RETRIEVAL_K", DEFAULT_FINAL_RETRIEVAL_K)
    context_budget = max_context_chars or env_int("RAG_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)

    grouped_by_source: dict[str, list[dict]] = {}
    for item in metadata:
        grouped_by_source.setdefault(item["source_name"], []).append(item)

    representatives: list[RetrievedChunk] = []
    round_index = 0
    source_names = sorted(grouped_by_source.keys())
    while len(representatives) < final_k and source_names:
        added_in_round = False
        for source_name in source_names:
            source_items = grouped_by_source[source_name]
            if round_index >= len(source_items):
                continue
            chunk_data = source_items[round_index]
            representatives.append(
                RetrievedChunk(
                    chunk_id=chunk_data["chunk_id"],
                    source_name=chunk_data["source_name"],
                    source_path=chunk_data["source_path"],
                    text=chunk_data["text"],
                    start_char=chunk_data["start_char"],
                    end_char=chunk_data["end_char"],
                    score=1.0 - (0.02 * round_index),
                    retrieval_score=1.0 - (0.02 * round_index),
                    page_number=chunk_data.get("page_number"),
                    section_heading=chunk_data.get("section_heading"),
                    source_variants=chunk_data.get("source_variants", []),
                )
            )
            added_in_round = True
            if len(representatives) >= final_k:
                break
        if not added_in_round:
            break
        round_index += 1

    return enforce_context_budget(representatives, final_k=final_k, max_context_chars=context_budget)[:final_k]
