from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import (
    answer_question,
    build_history_context,
    classify_answer_mode,
    classify_query_style,
    rewrite_query_with_history,
)
from src.retrieve import CorpusState, RetrievedChunk
from src.settings import RuntimeSettings


def make_chunk(chunk_id: str, score: float, text: str = "Chunk text") -> RetrievedChunk:
    """Create a minimal retrieved chunk for hybrid-answer tests."""

    return RetrievedChunk(
        chunk_id=chunk_id,
        source_name="course.pdf",
        source_path="data/docs/course.pdf",
        text=text,
        start_char=0,
        end_char=len(text),
        score=score,
        retrieval_score=score,
        page_number=1,
        section_heading="Section",
        source_variants=["course.pdf"],
    )


class HybridChatbotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        os.environ["LLM_PROVIDER"] = "test"
        os.environ["RAG_ENABLE_GENERAL_FALLBACK"] = "true"
        os.environ["RAG_MIN_SIMILARITY_SCORE"] = "0.65"
        os.environ["RAG_MIN_RELEVANT_CHUNKS"] = "2"
        os.environ["RAG_CHAT_HISTORY_TURNS"] = "4"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_classify_answer_mode_returns_documents_for_strong_retrieval(self) -> None:
        chunks = [
            make_chunk("a", 0.82, "Strong matching chunk"),
            make_chunk("b", 0.73, "Second strong matching chunk"),
        ]

        self.assertEqual(classify_answer_mode(chunks), "documents")

    def test_classify_answer_mode_returns_mixed_for_partial_retrieval(self) -> None:
        chunks = [
            make_chunk("a", 0.58, "Partially relevant chunk"),
            make_chunk("b", 0.31, "Weak chunk"),
        ]

        self.assertEqual(classify_answer_mode(chunks), "mixed")

    def test_answer_question_returns_general_knowledge_when_no_chunks_found(self) -> None:
        corpus_state = CorpusState(
            documents_count=1,
            chunks_indexed=10,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )
        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_chunks",
            return_value=[],
        ):
            response = answer_question("What is cache coherence?")

        self.assertEqual(response.answer_source, "general_knowledge")
        self.assertEqual(response.sources, [])
        self.assertIn("couldn't find strongly relevant passages", response.answer)

    def test_answer_question_keeps_previous_refusal_when_fallback_disabled(self) -> None:
        os.environ["RAG_ENABLE_GENERAL_FALLBACK"] = "false"
        corpus_state = CorpusState(
            documents_count=1,
            chunks_indexed=10,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_chunks",
            return_value=[],
        ), patch(
            "src.chatbot.is_test_mode",
            return_value=False,
        ):
            response = answer_question("What is cache coherence?")

        self.assertEqual(response.answer_source, "documents")
        self.assertEqual(
            response.answer,
            "I found uploaded documents, but I couldn't find strongly relevant passages for this question.",
        )

    def test_answer_question_returns_mixed_answer_with_document_sources(self) -> None:
        chunks = [
            make_chunk("a", 0.60, "The slides mention pipelining tradeoffs."),
            make_chunk("b", 0.40, "Another nearby chunk."),
        ]
        corpus_state = CorpusState(
            documents_count=1,
            chunks_indexed=10,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_chunks",
            return_value=chunks,
        ), patch(
            "src.chatbot.generate_grounded_answer",
            return_value="Document-grounded explanation.",
        ), patch(
            "src.chatbot.generate_general_knowledge_answer",
            return_value="General knowledge explanation.",
        ):
            response = answer_question("Why does pipelining improve throughput?")

        self.assertEqual(response.answer_source, "mixed")
        self.assertIn("Based on the uploaded materials:", response.answer)
        self.assertIn("Additional general context:", response.answer)
        self.assertEqual(len(response.sources), 1)

    def test_classify_query_style_detects_brief_factual_question(self) -> None:
        self.assertEqual(classify_query_style("how old is donald trump"), "brief")
        self.assertEqual(classify_query_style("Explain why pipelining improves throughput"), "normal")

    def test_brief_query_skips_rerank_when_enabled(self) -> None:
        chunks = [make_chunk("a", 0.72, "Short factual chunk")]
        corpus_state = CorpusState(
            documents_count=1,
            chunks_indexed=10,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_chunks",
            return_value=chunks,
        ) as mocked_retrieve, patch(
            "src.chatbot.generate_grounded_answer",
            return_value="Direct grounded answer.",
        ), patch(
            "src.chatbot.generate_general_knowledge_answer",
            return_value="Direct general answer.",
        ):
            response = answer_question("who invented linux")

        self.assertEqual(response.answer_source, "mixed")
        self.assertFalse(mocked_retrieve.call_args.kwargs["enable_rerank"])

    def test_rewrite_query_with_history_resolves_followup_reference(self) -> None:
        history = [
            {"role": "user", "content": "who is Donald Trump?"},
            {"role": "assistant", "content": "Donald Trump is an American politician and businessman."},
        ]

        rewritten = rewrite_query_with_history("could you give more info about him?", history)

        self.assertEqual(rewritten, "Could you give more info about Donald Trump?")

    def test_build_history_context_limits_recent_messages(self) -> None:
        history = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"},
        ]

        os.environ["RAG_CHAT_HISTORY_TURNS"] = "2"
        rendered = build_history_context(history)

        self.assertNotIn("Question 1", rendered)
        self.assertIn("Question 2", rendered)
        self.assertIn("Answer 3", rendered)

    def test_answer_question_uses_rewritten_query_for_retrieval(self) -> None:
        history = [
            {"role": "user", "content": "who is Donald Trump?"},
            {"role": "assistant", "content": "Donald Trump is an American politician and businessman."},
        ]
        corpus_state = CorpusState(
            documents_count=1,
            chunks_indexed=10,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_chunks",
            return_value=[],
        ) as mocked_retrieve:
            answer_question("could you give more info about him?", chat_history=history)

        self.assertEqual(
            mocked_retrieve.call_args.kwargs["query"],
            "Could you give more info about Donald Trump?",
        )

    def test_answer_question_uses_runtime_settings_for_retrieval_controls(self) -> None:
        settings = RuntimeSettings(
            provider="test",
            retrieval_k=7,
            final_retrieval_k=3,
            rerank_enabled=False,
            general_fallback_enabled=True,
        )
        corpus_state = CorpusState(
            documents_count=1,
            chunks_indexed=10,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_chunks",
            return_value=[],
        ) as mocked_retrieve:
            answer_question("What is cache coherence?", settings=settings)

        self.assertEqual(mocked_retrieve.call_args.kwargs["top_k"], 3)
        self.assertEqual(mocked_retrieve.call_args.kwargs["initial_k_override"], 7)
        self.assertFalse(mocked_retrieve.call_args.kwargs["enable_rerank"])

    def test_answer_question_reports_no_documents_only_when_corpus_missing(self) -> None:
        corpus_state = CorpusState(
            documents_count=0,
            chunks_indexed=0,
            has_documents=False,
            has_index=False,
            has_indexed_corpus=False,
        )

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state):
            response = answer_question("What is cache coherence?")

        self.assertIn("No uploaded course materials are available yet.", response.answer)

    def test_corpus_summary_queries_use_overview_retrieval(self) -> None:
        corpus_state = CorpusState(
            documents_count=5,
            chunks_indexed=109,
            has_documents=True,
            has_index=True,
            has_indexed_corpus=True,
        )
        overview_chunks = [make_chunk("a", 1.0, "Representative overview chunk")]

        with patch("src.chatbot.get_corpus_state", return_value=corpus_state), patch(
            "src.chatbot.retrieve_corpus_overview_chunks",
            return_value=overview_chunks,
        ) as mocked_overview, patch(
            "src.chatbot.generate_grounded_answer",
            return_value="Overview answer.",
        ):
            response = answer_question("Summarize the main ideas from the uploaded course materials")

        self.assertEqual(response.answer_source, "documents")
        mocked_overview.assert_called_once()
        self.assertEqual(response.answer, "Overview answer.")


if __name__ == "__main__":
    unittest.main()
