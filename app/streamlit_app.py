from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_shared import apply_page_styles, get_index_status, get_runtime_settings, initialize_session_state
from src.chatbot import answer_question, get_provider_label, is_ollama_mode, is_test_mode
from src.ingest import load_documents
from src.utils import truncate_text


st.set_page_config(page_title="Custom Data Chatbot", layout="wide")


def render_status_row(documents_count: int, chunks_indexed: int, model_name: str) -> None:
    """Render lightweight chat-first status chips."""

    settings = get_runtime_settings()
    chips = [
        f"<div class='status-chip'><strong>{documents_count}</strong> PDFs indexed</div>",
        f"<div class='status-chip'><strong>{chunks_indexed}</strong> chunks ready</div>",
        f"<div class='status-chip'><strong>{get_provider_label(settings)}</strong> {model_name}</div>",
        f"<div class='status-chip'><strong>{'Ready' if chunks_indexed else 'Waiting'}</strong> index status</div>",
    ]
    st.markdown(f"<div class='chip-row'>{''.join(chips)}</div>", unsafe_allow_html=True)

    if is_test_mode(settings):
        st.warning("Running in TEST MODE - OpenAI API disabled.")
    elif is_ollama_mode(settings):
        st.info("Using a local Ollama model for answer generation. This mode is not available on Streamlit Community Cloud.")


def render_sidebar(documents: list[Any], index_status: dict[str, Any]) -> None:
    """Render a minimal sidebar with status and recent prompts."""

    with st.sidebar:
        settings = get_runtime_settings()
        model_name = settings.ollama_model if settings.normalized_provider() == "ollama" else settings.openai_model
        st.markdown("### Custom Data Chatbot")
        st.caption("Chat-first RAG workspace")
        st.markdown("<div class='sidebar-section-title'>Status</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='sidebar-history-item'><strong>{get_provider_label(settings)}</strong><br>{model_name}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='sidebar-history-item'>"
                f"PDFs: <strong>{len(documents)}</strong><br>"
                f"Chunks: <strong>{index_status['chunks_indexed']}</strong></div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-history-item'>Index: <strong>{'Ready' if index_status['is_ready'] else 'Not built'}</strong></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-section-title'>Pages</div>", unsafe_allow_html=True)
        st.caption("Use Streamlit navigation to open `Workspace` or `Settings`.")

        if st.session_state["chat_history"]:
            st.markdown("<div class='sidebar-section-title'>Recent Chats</div>", unsafe_allow_html=True)
            recent_user_messages = [
                item["content"] for item in st.session_state["chat_history"] if item.get("role") == "user"
            ][-4:]
            for message in reversed(recent_user_messages):
                st.markdown(
                    f"<div class='sidebar-history-item'>{truncate_text(message, max_length=78)}</div>",
                    unsafe_allow_html=True,
                )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()


def render_empty_state(index_ready: bool) -> str | None:
    """Render a modern hero empty state and return a clicked prompt if any."""

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Custom course-material assistant</div>
            <h1 class="hero-title">Ask your documents anything.</h1>
            <p class="hero-subtitle">
                Search across uploaded PDFs, get grounded answers when your materials cover the topic,
                and fall back gracefully when they do not. Keep the conversation focused here; manage files
                in <strong>Workspace</strong> and models in <strong>Settings</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Summarize the main ideas from the uploaded course materials.",
        "What concepts appear most often across the PDFs?",
        "Explain the most important topic in simple terms.",
    ]
    cols = st.columns(len(suggestions))
    selected_prompt = None
    for col, prompt in zip(cols, suggestions):
        with col:
            if st.button(prompt, use_container_width=True, disabled=not index_ready):
                selected_prompt = prompt

    if not index_ready:
        st.caption("Build the index from the Workspace page before starting the conversation.")
    else:
        st.caption("Try a suggested prompt or type your own question below.")
    return selected_prompt


def render_sources(sources: list[Any]) -> None:
    """Render retrieved chunk details below an assistant answer."""

    if not sources:
        st.caption("No supporting chunks were retrieved.")
        return

    with st.expander("Supporting source chunks", expanded=False):
        for index, source in enumerate(sources, start=1):
            details = [f"Source {index}: {source.source_name}", f"Score {source.score:.3f}"]
            if getattr(source, "page_number", None) is not None:
                details.append(f"Page {source.page_number}")
            if getattr(source, "section_heading", None):
                details.append(f"Section {source.section_heading}")
            meta = " | ".join(details)
            st.markdown(
                (
                    "<div class='source-card'>"
                    f"<div class='source-meta'>{meta}</div>"
                    f"<div>{truncate_text(source.text, max_length=700)}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if getattr(source, "source_variants", None):
                st.caption(f"Canonicalized variants: {', '.join(source.source_variants)}")


def get_answer_source_label(answer_source: str) -> str:
    """Return a user-facing answer source label."""

    labels = {
        "documents": "Uploaded materials",
        "general_knowledge": "General knowledge",
        "mixed": "Mixed (documents + general knowledge)",
    }
    return labels.get(answer_source, "Unknown")


def render_answer_source(answer_source: str) -> None:
    """Render a small badge that explains where the answer came from."""

    st.markdown(
        f"<div class='answer-source-badge'>Answer source: {get_answer_source_label(answer_source)}</div>",
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Render the stored chat history in chat-message format."""

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_source(message.get("answer_source", "documents"))
            st.markdown(message["content"])
            if message["role"] == "assistant":
                answer_source = message.get("answer_source", "documents")
                if answer_source == "general_knowledge":
                    corpus_state = message.get("corpus_state")
                    if corpus_state and not corpus_state.get("has_documents", False):
                        st.caption("No uploaded course materials are available yet.")
                    elif corpus_state and corpus_state.get("has_documents", False):
                        st.caption("Uploaded documents exist, but no strongly relevant passages were found for this question.")
                    else:
                        st.caption("No relevant information was found in the uploaded documents.")
                else:
                    render_sources(message.get("sources", []))


def append_chat_exchange(question: str) -> None:
    """Answer a question and append both user and assistant messages."""

    prior_history = list(st.session_state["chat_history"])
    settings = get_runtime_settings()
    st.session_state["chat_history"].append({"role": "user", "content": question})

    response = answer_question(question, chat_history=prior_history, settings=settings)
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": response.answer,
            "answer_source": response.answer_source,
            "sources": response.sources,
            "corpus_state": response.corpus_state.__dict__ if response.corpus_state else None,
        }
    )


def main() -> None:
    """Run the Streamlit application."""

    apply_page_styles()
    initialize_session_state()

    documents = load_documents()
    index_status = get_index_status()
    settings = get_runtime_settings()
    model_name = settings.ollama_model if settings.normalized_provider() == "ollama" else settings.openai_model

    render_sidebar(documents, index_status)
    render_status_row(
        documents_count=len(documents),
        chunks_indexed=index_status["chunks_indexed"],
        model_name=model_name,
    )

    st.title("Chat")
    st.caption("Conversation-first workspace for your uploaded PDFs.")

    selected_prompt = None
    if not st.session_state["chat_history"]:
        selected_prompt = render_empty_state(index_ready=index_status["is_ready"])

    render_chat_history()

    user_prompt = selected_prompt or st.chat_input(
        "Ask a question about your documents",
        disabled=not index_status["is_ready"],
    )
    if user_prompt:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                append_chat_exchange(user_prompt)
            st.rerun()
        except Exception as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
