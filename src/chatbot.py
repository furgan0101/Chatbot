from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from textwrap import shorten
from typing import Any, Literal

import ollama
from openai import OpenAI

from src.retrieve import CorpusState, RetrievedChunk, get_corpus_state, retrieve_chunks, retrieve_corpus_overview_chunks
from src.settings import RuntimeSettings, default_runtime_settings
from src.utils import env_flag, env_float, env_int

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
FALLBACK_MESSAGE = "I do not know based on the provided documents."
NO_DOCUMENTS_MESSAGE = "No uploaded course materials are available yet."
NO_RELEVANT_MATCH_MESSAGE = "I found uploaded documents, but I couldn't find strongly relevant passages for this question."
SUPPORTED_PROVIDERS = {"test", "ollama", "openai"}
DEFAULT_FINAL_RETRIEVAL_K = 4
DEFAULT_SOURCE_DISPLAY_K = 5
DEFAULT_ANSWER_STYLE = os.getenv("RAG_DEFAULT_ANSWER_STYLE", "student-friendly, 1-3 paragraphs")
DEFAULT_ENABLE_GENERAL_FALLBACK = True
DEFAULT_MIN_SIMILARITY_SCORE = 0.65
DEFAULT_MIN_RELEVANT_CHUNKS = 2
DEFAULT_MIXED_MODE_MARGIN = 0.15
DEFAULT_BRIEF_QUERY_MAX_WORDS = 8
DEFAULT_BRIEF_NUM_PREDICT = 80
DEFAULT_NORMAL_NUM_PREDICT = 220
DEFAULT_SKIP_RERANK_FOR_BRIEF_QUERIES = True
DEFAULT_CHAT_HISTORY_TURNS = 4
HIGH_STAKES_TERMS = {
    "medical",
    "medicine",
    "medication",
    "diagnosis",
    "treatment",
    "legal",
    "law",
    "lawsuit",
    "financial",
    "finance",
    "investment",
    "tax",
    "prescription",
    "emergency",
}

AnswerSource = Literal["documents", "general_knowledge", "mixed"]
AnswerStyle = Literal["brief", "normal"]


@dataclass
class ChatbotResponse:
    """A grounded chatbot response with supporting retrieved chunks."""

    answer: str
    answer_source: AnswerSource
    sources: list[RetrievedChunk]
    corpus_state: CorpusState | None = None


def get_recent_chat_history(
    chat_history: list[dict[str, Any]] | None,
    turns: int | None = None,
) -> list[dict[str, Any]]:
    """Return only the recent chat messages used for conversational context."""

    if not chat_history:
        return []

    max_turns = turns or env_int("RAG_CHAT_HISTORY_TURNS", DEFAULT_CHAT_HISTORY_TURNS)
    max_messages = max(0, max_turns * 2)
    return chat_history[-max_messages:]


def build_history_context(chat_history: list[dict[str, Any]] | None) -> str:
    """Format recent chat history for answer-generation prompts."""

    recent_history = get_recent_chat_history(chat_history)
    if not recent_history:
        return ""

    lines: list[str] = []
    for message in recent_history:
        role = message.get("role", "user").strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def resolve_runtime_settings(settings: RuntimeSettings | None = None) -> RuntimeSettings:
    """Return explicit runtime settings or env-backed defaults."""

    if settings is not None:
        return settings

    resolved = default_runtime_settings()
    if env_flag("TEST_MODE", default=False):
        resolved.provider = "test"
    return resolved


def get_provider_label(settings: RuntimeSettings | None = None) -> str:
    """Return a UI-friendly provider label."""

    provider = resolve_runtime_settings(settings).normalized_provider()
    labels = {
        "test": "Test",
        "ollama": "Ollama",
        "openai": "OpenAI",
    }
    return labels[provider]


def is_test_mode(settings: RuntimeSettings | None = None) -> bool:
    """Return True when the project should avoid external LLM calls."""

    return resolve_runtime_settings(settings).normalized_provider() == "test"


def is_ollama_mode(settings: RuntimeSettings | None = None) -> bool:
    """Return True when Ollama is the active provider."""

    return resolve_runtime_settings(settings).normalized_provider() == "ollama"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format all selected chunks into a prompt-friendly evidence block."""

    if not chunks:
        return ""

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata_lines = [
            f"Source {index}: {chunk.source_name}",
            f"Chunk ID: {chunk.chunk_id}",
        ]
        if chunk.page_number is not None:
            metadata_lines.append(f"Page: {chunk.page_number}")
        if chunk.section_heading:
            metadata_lines.append(f"Section: {chunk.section_heading}")
        if chunk.source_variants:
            metadata_lines.append(f"Canonicalized file variants: {', '.join(chunk.source_variants)}")
        metadata_lines.append(f"Text: {chunk.text}")
        parts.append("\n".join(metadata_lines))

    return "\n\n".join(parts)


def mock_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Create a deterministic offline response from multiple retrieved chunks."""

    if not chunks:
        return (
            "Mock Response (API disabled)\n\n"
            f"Question: {question}\n\n"
            "Relevant context found in documents:\n"
            "No relevant context was retrieved.\n\n"
            f"Summary:\n{FALLBACK_MESSAGE}"
        )

    context_lines: list[str] = []
    summary_points: list[str] = []

    for index, chunk in enumerate(chunks[:4], start=1):
        sanitized_text = "".join(
            character for character in chunk.text if character.isprintable() or character.isspace()
        )
        preview = shorten(" ".join(sanitized_text.split()), width=320, placeholder="...")
        location = f"page {chunk.page_number}" if chunk.page_number is not None else "document scope"
        context_lines.append(f"{index}. [{chunk.source_name}, {location}] {preview}")
        summary_points.append(
            f"Chunk {index} shows that {preview.lower()}"
        )

    return (
        "Mock Response (API disabled)\n\n"
        f"Question: {question}\n\n"
        "Relevant context found in documents:\n"
        f"{chr(10).join(context_lines)}\n\n"
        "Summary:\n"
        "Based on several retrieved sections, "
        + " ".join(summary_points)
    )


def mock_general_answer(question: str) -> str:
    """Create a deterministic offline general-knowledge answer."""

    style = classify_query_style(question)
    if style == "brief":
        return "Mock Response (API disabled)\n\nA short direct answer would be returned here."
    return (
        "Mock Response (API disabled)\n\n"
        f"Question: {question}\n\n"
        "A concise general-knowledge explanation would be returned here."
    )


def build_document_prompt(answer_style: AnswerStyle) -> str:
    """Prompt for fully document-grounded answers."""

    if answer_style == "brief":
        return (
            "You are a course-material QA assistant. Answer using only the retrieved PDF context. "
            "Use recent conversation only to resolve references like 'him', 'it', or 'that'. "
            "Give a direct answer in 1-2 sentences when the question is simple and factual. "
            "If the retrieved material is incomplete, say that briefly. Do not add filler, meta commentary, or claims beyond the context."
        )

    return (
        "You are a course-material QA assistant. Answer the user's question using only the retrieved PDF context. "
        "Use recent conversation only to resolve follow-up references and preserve continuity. "
        "Synthesize information across the provided chunks instead of relying on a single sentence. Write a clear, "
        "student-friendly explanation in 1-2 short paragraphs. If multiple chunks support the same idea, combine them into "
        "a single coherent explanation. Do not copy text verbatim and do not make claims beyond the retrieved material. "
        "If the material is incomplete, clearly say so."
    )


def build_general_knowledge_prompt(question: str, answer_style: AnswerStyle) -> str:
    """Prompt for fallback answers that rely on model knowledge."""

    if answer_style == "brief":
        prompt = (
            "Answer from general knowledge in a natural chatbot style. "
            "Use recent conversation to resolve follow-up references when needed. "
            "For simple factual questions, reply in 1-2 sentences max. "
            "Give the answer directly. Do not repeat disclaimers, source notes, or meta discussion."
        )
    else:
        prompt = (
            "Answer from general knowledge in a natural chatbot style. "
            "Use recent conversation to resolve follow-up references when needed. "
            "For broader explanatory questions, reply in 1-2 short paragraphs. "
            "Keep the answer direct and useful. Avoid repeated disclaimers, reliability lectures, and meta commentary."
        )

    if is_high_stakes_question(question):
        prompt += " Add cautionary wording only if the topic is genuinely high-stakes and uncertainty matters."

    return prompt


def is_corpus_summary_query(question: str) -> bool:
    """Return True for broad corpus-level questions that need representative retrieval."""

    normalized = query_normalize(question).lower()
    corpus_patterns = (
        "uploaded course materials",
        "uploaded documents",
        "these pdfs",
        "the pdfs",
        "the documents",
        "course materials",
    )
    summary_patterns = (
        "summarize",
        "summary",
        "main ideas",
        "topics are covered",
        "what topics are covered",
        "overview",
        "what is covered",
    )
    return any(summary in normalized for summary in summary_patterns) and any(
        corpus_term in normalized for corpus_term in corpus_patterns
    )


def prefix_general_answer(note: str | None, answer: str) -> str:
    """Prepend a short, accurate note before a fallback answer when needed."""

    if not note:
        return answer
    return f"{note}\n\n{answer}"


def looks_like_standalone_question(question: str) -> bool:
    """Return True when the message already names its topic clearly."""

    normalized = query_normalize(question)
    if not normalized:
        return True

    pronoun_markers = {
        "he",
        "she",
        "him",
        "her",
        "they",
        "them",
        "it",
        "its",
        "that",
        "this",
        "those",
        "these",
        "former",
        "latter",
    }
    tokens = re.findall(r"[a-z0-9']+", normalized.lower())
    if not tokens:
        return True
    if any(token in pronoun_markers for token in tokens):
        return False

    followup_openers = (
        "tell me more",
        "more info",
        "more information",
        "what about",
        "and what about",
        "could you give more info",
        "can you give more info",
        "expand on",
        "go deeper",
    )
    lowered = normalized.lower()
    return not any(lowered.startswith(opener) for opener in followup_openers)


def needs_query_rewrite(question: str, chat_history: list[dict[str, Any]] | None) -> bool:
    """Return True when the current message likely depends on previous turns."""

    if not get_recent_chat_history(chat_history):
        return False
    return not looks_like_standalone_question(question)


def extract_topic_candidate(text: str) -> str | None:
    """Extract a likely topic phrase from a prior user or assistant message."""

    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None

    explicit_patterns = [
        r"(?i)^(?:who|what|when|where)\s+(?:is|was|are|were)\s+(.+?)[\?\.]?$",
        r"(?i)^(?:tell me about|explain|define|describe)\s+(.+?)[\?\.]?$",
        r"(?i)^(?:could you explain|can you explain)\s+(.+?)[\?\.]?$",
    ]
    for pattern in explicit_patterns:
        match = re.match(pattern, cleaned)
        if match:
            candidate = match.group(1).strip(" .?!,")
            if candidate:
                return candidate

    capitalized_phrases = re.findall(r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b", cleaned)
    for phrase in reversed(capitalized_phrases):
        if len(phrase) > 1:
            return phrase.strip()

    noun_like_patterns = [
        r"(?i)\b(?:about|on|regarding)\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,})",
        r"(?i)\b([a-zA-Z][a-zA-Z0-9\-\s]{2,})\s+(?:is|are|was|were)\b",
    ]
    for pattern in noun_like_patterns:
        match = re.search(pattern, cleaned)
        if match:
            candidate = match.group(1).strip(" .?!,")
            if candidate:
                return candidate

    return None


def resolve_followup_topic(chat_history: list[dict[str, Any]] | None) -> str | None:
    """Find the most recent explicit topic from recent chat history."""

    recent_history = get_recent_chat_history(chat_history)
    prioritized_history = [
        message for message in reversed(recent_history) if str(message.get("role", "")).lower() == "user"
    ] + [
        message for message in reversed(recent_history) if str(message.get("role", "")).lower() != "user"
    ]

    for message in prioritized_history:
        candidate = extract_topic_candidate(str(message.get("content", "")))
        if candidate:
            return candidate
    return None


def rewrite_query_with_history(
    question: str,
    chat_history: list[dict[str, Any]] | None,
) -> str:
    """Rewrite a follow-up question into a standalone query when needed."""

    normalized_question = query_normalize(question)
    if not needs_query_rewrite(normalized_question, chat_history):
        return normalized_question

    topic = resolve_followup_topic(chat_history)
    if not topic:
        return normalized_question

    rewritten = normalized_question
    replacement_patterns = [
        (r"(?i)\bhim\b", topic),
        (r"(?i)\bher\b", topic),
        (r"(?i)\bthem\b", topic),
        (r"(?i)\bit\b", topic),
        (r"(?i)\bthat\b", topic),
        (r"(?i)\bthis\b", topic),
        (r"(?i)\bthe previous topic\b", topic),
    ]
    for pattern, replacement in replacement_patterns:
        rewritten = re.sub(pattern, replacement, rewritten)

    lowered = rewritten.lower()
    if lowered.startswith(("tell me more", "more info", "more information", "could you give more info", "can you give more info")):
        return f"Could you give more info about {topic}?"
    if lowered.startswith(("what about", "and what about")):
        return re.sub(r"(?i)^(and\s+)?what about\b", f"What about {topic}", rewritten)
    if lowered == normalized_question.lower():
        return f"{normalized_question} about {topic}"
    return rewritten


def classify_query_style(question: str) -> AnswerStyle:
    """Classify whether the user likely wants a brief factual answer."""

    normalized = query_normalize(question).lower()
    words = re.findall(r"[a-z0-9']+", normalized)
    if not words:
        return "normal"

    max_words = env_int("RAG_BRIEF_QUERY_MAX_WORDS", DEFAULT_BRIEF_QUERY_MAX_WORDS)
    question_starters = {"who", "what", "when", "where", "which", "how"}
    brief_patterns = (
        normalized.startswith("who "),
        normalized.startswith("what "),
        normalized.startswith("when "),
        normalized.startswith("where "),
        normalized.startswith("which "),
        normalized.startswith("how old "),
        normalized.startswith("define "),
    )

    if len(words) <= max_words and (words[0] in question_starters or any(brief_patterns)):
        return "brief"
    return "normal"


def is_high_stakes_question(question: str) -> bool:
    """Return True when the question appears to be about a high-stakes topic."""

    normalized = query_normalize(question).lower()
    return any(term in normalized for term in HIGH_STAKES_TERMS)


def get_generation_limit_for_settings(
    answer_style: AnswerStyle,
    settings: RuntimeSettings | None = None,
) -> int:
    """Return the active generation budget, preferring runtime settings when provided."""

    runtime_settings = resolve_runtime_settings(settings)
    configured_limit = max(1, int(runtime_settings.max_tokens))
    if answer_style == "brief":
        return min(configured_limit, env_int("RAG_BRIEF_NUM_PREDICT", DEFAULT_BRIEF_NUM_PREDICT))
    return configured_limit


@lru_cache(maxsize=4)
def get_openai_client(api_key: str) -> OpenAI:
    """Create an OpenAI client using the API key from environment variables."""

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it in Streamlit secrets.")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=4)
def get_ollama_client(base_url: str):
    """Create a cached Ollama client for the configured host."""

    return ollama.Client(host=base_url)


def generate_openai_answer(
    settings: RuntimeSettings,
    question: str,
    instructions: str,
    context: str | None = None,
    history_context: str | None = None,
    max_output_tokens: int = DEFAULT_NORMAL_NUM_PREDICT,
) -> str:
    """Generate an answer using OpenAI."""

    client = get_openai_client(settings.openai_api_key.strip())
    user_input = f"Question:\n{question}"
    if history_context:
        user_input = f"Recent conversation:\n{history_context}\n\n{user_input}"
    if context:
        user_input += f"\n\nRetrieved context:\n{context}"
    response = client.responses.create(
        model=settings.openai_model.strip() or DEFAULT_OPENAI_MODEL,
        instructions=instructions,
        input=user_input,
        max_output_tokens=max_output_tokens,
        temperature=float(settings.temperature),
        top_p=float(settings.top_p),
    )
    return response.output_text.strip()


def generate_ollama_answer(
    settings: RuntimeSettings,
    question: str,
    instructions: str,
    context: str | None = None,
    history_context: str | None = None,
    num_predict: int = DEFAULT_NORMAL_NUM_PREDICT,
) -> str:
    """Generate an answer using a local Ollama model."""

    prompt_parts = [instructions]
    if history_context:
        prompt_parts.append(f"Recent conversation:\n{history_context}")
    prompt_parts.append(f"Question:\n{question}")
    if context:
        prompt_parts.append(f"Retrieved context:\n{context}")
    prompt = "\n\n".join(prompt_parts)

    try:
        client = get_ollama_client(settings.ollama_base_url.strip() or "http://localhost:11434")
        response = client.chat(
            model=settings.ollama_model.strip() or DEFAULT_OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": num_predict,
                "temperature": float(settings.temperature),
                "top_p": float(settings.top_p),
            },
        )
    except Exception as error:
        message = str(error).lower()
        if "not found" in message or "model" in message and "pull" in message:
            raise RuntimeError(
                "Ollama model is not available locally. Make sure the selected model is installed, "
                "for example by running: ollama run llama3"
            ) from error
        raise RuntimeError(
            "Ollama is not reachable. Make sure Ollama is installed, running, and the selected model is available locally."
        ) from error

    answer = response["message"]["content"].strip()
    return answer


def generate_grounded_answer(
    settings: RuntimeSettings,
    question: str,
    context: str,
    chunks: list[RetrievedChunk],
    answer_style: AnswerStyle,
    history_context: str | None = None,
) -> str:
    """Route document-grounded answer generation to the configured provider."""

    provider = settings.normalized_provider()
    if provider == "test":
        return mock_answer(question, chunks)
    if provider == "ollama":
        return generate_ollama_answer(
            settings=settings,
            question=question,
            instructions=build_document_prompt(answer_style),
            context=context,
            history_context=history_context,
            num_predict=get_generation_limit_for_settings(answer_style, settings),
        )
    if provider == "openai":
        return generate_openai_answer(
            settings=settings,
            question=question,
            instructions=build_document_prompt(answer_style),
            context=context,
            history_context=history_context,
            max_output_tokens=get_generation_limit_for_settings(answer_style, settings),
        )
    raise RuntimeError("Unsupported LLM provider configuration.")


def generate_general_knowledge_answer(
    settings: RuntimeSettings,
    question: str,
    answer_style: AnswerStyle,
    history_context: str | None = None,
) -> str:
    """Route general-knowledge fallback generation to the configured provider."""

    provider = settings.normalized_provider()
    if provider == "test":
        return mock_general_answer(question)
    if provider == "ollama":
        return generate_ollama_answer(
            settings=settings,
            question=question,
            instructions=build_general_knowledge_prompt(question, answer_style),
            context=None,
            history_context=history_context,
            num_predict=get_generation_limit_for_settings(answer_style, settings),
        )
    if provider == "openai":
        return generate_openai_answer(
            settings=settings,
            question=question,
            instructions=build_general_knowledge_prompt(question, answer_style),
            context=None,
            history_context=history_context,
            max_output_tokens=get_generation_limit_for_settings(answer_style, settings),
        )
    raise RuntimeError("Unsupported LLM provider configuration.")


def get_relevance_config() -> tuple[float, int]:
    """Return the configurable thresholds used to decide answer source mode."""

    min_similarity = env_float("RAG_MIN_SIMILARITY_SCORE", DEFAULT_MIN_SIMILARITY_SCORE)
    min_relevant_chunks = env_int("RAG_MIN_RELEVANT_CHUNKS", DEFAULT_MIN_RELEVANT_CHUNKS)
    return min_similarity, max(1, min_relevant_chunks)


def classify_answer_mode(chunks: list[RetrievedChunk]) -> AnswerSource:
    """Classify retrieval strength as documents, mixed, or general knowledge.

    The existing retrieval pipeline stays intact. This decision sits after
    retrieval/reranking and before generation so the app can transparently fall
    back when document evidence is weak.
    """

    if not chunks:
        return "general_knowledge"

    min_similarity, min_relevant_chunks = get_relevance_config()
    relevant_chunks = [chunk for chunk in chunks if chunk.score >= min_similarity]
    if chunks[0].score >= min_similarity and len(relevant_chunks) >= min_relevant_chunks:
        return "documents"

    partial_similarity = max(0.0, min_similarity - DEFAULT_MIXED_MODE_MARGIN)
    partial_chunks = [chunk for chunk in chunks if chunk.score >= partial_similarity]
    if partial_chunks:
        return "mixed"

    return "general_knowledge"


def compose_mixed_answer(document_answer: str, general_answer: str) -> str:
    """Merge document-grounded and general-knowledge sections into one response."""

    return (
        "Based on the uploaded materials:\n"
        f"{document_answer}\n\n"
        "Additional general context:\n"
        f"{general_answer}"
    )


def unique_sources_for_display(
    chunks: list[RetrievedChunk],
    max_sources: int | None = None,
) -> list[RetrievedChunk]:
    """Return a compact, source-transparent subset for the UI."""

    limit = max_sources or env_int("RAG_SOURCE_DISPLAY_K", DEFAULT_SOURCE_DISPLAY_K)
    selected: list[RetrievedChunk] = []
    seen_signatures: set[tuple[str, int | None, str | None]] = set()

    for chunk in chunks:
        signature = (chunk.source_name, chunk.page_number, chunk.section_heading)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(chunk)
        if len(selected) >= limit:
            break

    return selected


def answer_question(
    question: str,
    top_k: int | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    settings: RuntimeSettings | None = None,
) -> ChatbotResponse:
    """Retrieve context and return a grounded answer for the user question."""

    runtime_settings = resolve_runtime_settings(settings)
    corpus_state = get_corpus_state()
    normalized_question = query_normalize(question)
    history_context = build_history_context(chat_history)
    rewritten_question = rewrite_query_with_history(normalized_question, chat_history)
    answer_style = classify_query_style(rewritten_question)
    final_k = top_k or max(1, int(runtime_settings.final_retrieval_k))
    use_corpus_overview = is_corpus_summary_query(rewritten_question)
    fallback_enabled = runtime_settings.general_fallback_enabled

    if not corpus_state.has_documents:
        if fallback_enabled:
            return ChatbotResponse(
                answer=prefix_general_answer(
                    NO_DOCUMENTS_MESSAGE,
                    generate_general_knowledge_answer(
                        runtime_settings,
                        rewritten_question,
                        answer_style,
                        history_context=history_context,
                    ),
                ),
                answer_source="general_knowledge",
                sources=[],
                corpus_state=corpus_state,
            )
        if is_test_mode(runtime_settings):
            return ChatbotResponse(
                answer=NO_DOCUMENTS_MESSAGE,
                answer_source="documents",
                sources=[],
                corpus_state=corpus_state,
            )
        return ChatbotResponse(
            answer=NO_DOCUMENTS_MESSAGE,
            answer_source="documents",
            sources=[],
            corpus_state=corpus_state,
        )

    if not corpus_state.has_indexed_corpus:
        no_index_message = "Uploaded documents exist, but the retrieval index is not ready yet."
        return ChatbotResponse(
            answer=no_index_message,
            answer_source="documents",
            sources=[],
            corpus_state=corpus_state,
        )

    if use_corpus_overview:
        chunks = retrieve_corpus_overview_chunks(top_k=final_k)
    else:
        chunks = retrieve_chunks(
            query=rewritten_question,
            top_k=final_k,
            initial_k_override=max(1, int(runtime_settings.retrieval_k)),
            enable_rerank=runtime_settings.rerank_enabled and not (
                answer_style == "brief"
                and env_flag("RAG_SKIP_RERANK_FOR_BRIEF_QUERIES", DEFAULT_SKIP_RERANK_FOR_BRIEF_QUERIES)
            ),
        )

    if not chunks:
        if fallback_enabled:
            return ChatbotResponse(
                answer=prefix_general_answer(
                    NO_RELEVANT_MATCH_MESSAGE,
                    generate_general_knowledge_answer(
                        runtime_settings,
                        rewritten_question,
                        answer_style,
                        history_context=history_context,
                    ),
                ),
                answer_source="general_knowledge",
                sources=[],
                corpus_state=corpus_state,
            )
        if is_test_mode(runtime_settings):
            return ChatbotResponse(
                answer=NO_RELEVANT_MATCH_MESSAGE,
                answer_source="documents",
                sources=[],
                corpus_state=corpus_state,
            )
        return ChatbotResponse(
            answer=NO_RELEVANT_MATCH_MESSAGE,
            answer_source="documents",
            sources=[],
            corpus_state=corpus_state,
        )

    answer_mode = "documents" if use_corpus_overview and chunks else classify_answer_mode(chunks)
    if not fallback_enabled and answer_mode != "documents":
        if is_test_mode(runtime_settings):
            return ChatbotResponse(
                answer=NO_RELEVANT_MATCH_MESSAGE,
                answer_source="documents",
                sources=[],
                corpus_state=corpus_state,
            )
        return ChatbotResponse(
            answer=NO_RELEVANT_MATCH_MESSAGE,
            answer_source="documents",
            sources=[],
            corpus_state=corpus_state,
        )

    context = build_context(chunks)
    if answer_mode == "documents":
        answer = generate_grounded_answer(
            runtime_settings,
            question=rewritten_question,
            context=context,
            chunks=chunks,
            answer_style=answer_style,
            history_context=history_context,
        )
    elif answer_mode == "mixed":
        document_answer = generate_grounded_answer(
            runtime_settings,
            question=rewritten_question,
            context=context,
            chunks=chunks,
            answer_style=answer_style,
            history_context=history_context,
        )
        general_answer = generate_general_knowledge_answer(
            runtime_settings,
            rewritten_question,
            answer_style,
            history_context=history_context,
        )
        answer = compose_mixed_answer(document_answer=document_answer, general_answer=general_answer)
    else:
        answer = prefix_general_answer(
            NO_RELEVANT_MATCH_MESSAGE,
            generate_general_knowledge_answer(
                runtime_settings,
                rewritten_question,
                answer_style,
                history_context=history_context,
            ),
        )

    if not answer:
        answer = FALLBACK_MESSAGE

    return ChatbotResponse(
        answer=answer,
        answer_source=answer_mode,
        sources=unique_sources_for_display(chunks) if answer_mode != "general_knowledge" else [],
        corpus_state=corpus_state,
    )


def query_normalize(question: str) -> str:
    """Normalize user input before retrieval."""

    return " ".join(question.strip().split())
