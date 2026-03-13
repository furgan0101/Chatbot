from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TEST_MODE"] = "true"
os.environ["LLM_PROVIDER"] = "test"

from src.chatbot import mock_answer
from src.chunking import chunk_documents
from src.embed import build_faiss_index
from src.embed import generate_embeddings
from src.embed import load_embedding_model
from src.embed import save_index
from src.ingest import load_documents
from src.retrieve import retrieve_chunks


def main() -> None:
    """Run the offline RAG pipeline test from PDFs through mock answering."""

    test_question = "What is artificial intelligence?"

    documents = load_documents()
    print("Loaded documents")
    print(f"Document count: {len(documents)}")
    if not documents:
        print("No readable PDF documents were found in data/docs.")
        return

    chunks = chunk_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")
    if not chunks:
        print("No chunks were created from the loaded documents.")
        return

    model = load_embedding_model()
    embeddings = generate_embeddings([chunk.text for chunk in chunks], model)
    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    print(f"FAISS index size: {index.ntotal}")

    retrieved = retrieve_chunks(test_question, top_k=3, score_threshold=0.0)
    print("Retrieved chunks")
    if not retrieved:
        print("No chunks were retrieved for the sample question.")
    else:
        for item in retrieved:
            location = f"page={item.page_number}" if item.page_number is not None else "page=n/a"
            print(f"- {item.source_name} | {location} | score={item.score:.3f} | chunk_id={item.chunk_id}")
            print(f"  {item.text[:200]}...")

    response = mock_answer(test_question, retrieved)
    print("\nMock response")
    print(response)


if __name__ == "__main__":
    main()
