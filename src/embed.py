from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import TextChunk, chunk_documents
from src.ingest import load_documents_with_deduplication
from src.utils import (
    BUILD_REPORT_FILE,
    INDEX_FILE,
    METADATA_FILE,
    env_flag,
    env_int,
    ensure_directories,
    save_json,
)


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 850
DEFAULT_CHUNK_OVERLAP = 120


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load and return the sentence-transformers embedding model."""

    return SentenceTransformer(
        model_name,
        local_files_only=env_flag("TEST_MODE", default=False),
    )


def generate_embeddings(
    texts: Iterable[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Generate normalized embeddings for a list of texts."""

    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product FAISS index for normalized embeddings."""

    if embeddings.size == 0:
        raise ValueError("Cannot build a FAISS index from empty embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: List[TextChunk]) -> None:
    """Persist the FAISS index and chunk metadata to disk."""

    ensure_directories()
    faiss.write_index(index, str(INDEX_FILE))
    save_json([chunk.to_dict() for chunk in chunks], METADATA_FILE)
    _read_index.cache_clear()


@lru_cache(maxsize=2)
def _read_index(index_path: str, modified_time_ns: int) -> faiss.Index:
    """Read a FAISS index, keyed by file path and modification time for cache safety."""

    del modified_time_ns
    return faiss.read_index(index_path)


def load_index() -> faiss.Index:
    """Load the FAISS index from disk."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}.")
    stat = Path(INDEX_FILE).stat()
    return _read_index(str(INDEX_FILE), stat.st_mtime_ns)


def rebuild_vector_store(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Read PDFs, deduplicate, chunk, embed, and save a fresh FAISS index."""

    resolved_chunk_size = chunk_size or env_int("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    resolved_chunk_overlap = chunk_overlap or env_int("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)

    load_result = load_documents_with_deduplication(deduplicate=True)
    documents = load_result.documents
    duplicates = load_result.duplicates

    if not documents:
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
            _read_index.cache_clear()
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        save_json({"duplicates_skipped": [item.to_dict() for item in duplicates]}, BUILD_REPORT_FILE)
        return {
            "documents_indexed": 0,
            "chunks_indexed": 0,
            "duplicates_skipped": len(duplicates),
            "duplicate_records": [item.to_dict() for item in duplicates],
            "index_path": str(INDEX_FILE),
            "metadata_path": str(METADATA_FILE),
        }

    chunks = chunk_documents(
        documents=documents,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
    )
    if not chunks:
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
            _read_index.cache_clear()
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        save_json({"duplicates_skipped": [item.to_dict() for item in duplicates]}, BUILD_REPORT_FILE)
        return {
            "documents_indexed": len(documents),
            "chunks_indexed": 0,
            "duplicates_skipped": len(duplicates),
            "duplicate_records": [item.to_dict() for item in duplicates],
            "index_path": str(INDEX_FILE),
            "metadata_path": str(METADATA_FILE),
        }

    model = load_embedding_model()
    embeddings = generate_embeddings([chunk.text for chunk in chunks], model)
    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    build_report = {
        "documents_indexed": len(documents),
        "chunks_indexed": len(chunks),
        "duplicates_skipped": [item.to_dict() for item in duplicates],
        "config": {
            "chunk_size": resolved_chunk_size,
            "chunk_overlap": resolved_chunk_overlap,
        },
    }
    save_json(build_report, BUILD_REPORT_FILE)

    for duplicate in duplicates:
        print(
            "Skipped duplicate PDF:",
            duplicate.skipped_source_name,
            "->",
            duplicate.canonical_source_name,
            f"({duplicate.duplicate_type}, similarity={duplicate.similarity:.3f})",
        )

    return {
        "documents_indexed": len(documents),
        "chunks_indexed": len(chunks),
        "duplicates_skipped": len(duplicates),
        "duplicate_records": [item.to_dict() for item in duplicates],
        "index_path": str(INDEX_FILE),
        "metadata_path": str(METADATA_FILE),
    }
