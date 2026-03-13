from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_shared import apply_page_styles, get_runtime_settings, initialize_session_state
from src.chatbot import get_provider_label
from src.settings import RuntimeSettings, load_runtime_settings, persist_runtime_settings


st.set_page_config(page_title="Settings", layout="wide")


def render_settings_page() -> None:
    """Render a dedicated settings page for provider and RAG controls."""

    settings = get_runtime_settings()

    st.title("Settings")
    st.write("Choose the model provider, fine-tune generation behavior, and adjust the RAG pipeline for this session.")

    summary = st.columns(3)
    summary[0].metric("Active Provider", get_provider_label(settings))
    summary[1].metric("Generation Limit", int(settings.max_tokens))
    summary[2].metric("Initial Retrieval K", int(settings.retrieval_k))

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    provider_label = st.selectbox(
        "Provider",
        options=["Ollama", "OpenAI"],
        index=0 if settings.normalized_provider() == "ollama" else 1,
    )
    provider = provider_label.lower()

    if provider == "ollama":
        st.warning("Ollama requires a local service and is not available on Streamlit Community Cloud. Use OpenAI for cloud deployment.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Provider")
        ollama_base_url = settings.ollama_base_url
        ollama_model = settings.ollama_model
        openai_api_key = settings.openai_api_key
        openai_model = settings.openai_model

        if provider == "ollama":
            ollama_base_url = st.text_input("Ollama base URL", value=settings.ollama_base_url)
            ollama_model = st.text_input("Ollama model", value=settings.ollama_model)
        else:
            openai_api_key = st.text_input(
                "OpenAI API key",
                value=settings.openai_api_key,
                type="password",
            )
            openai_model = st.text_input("OpenAI model", value=settings.openai_model)

    with right:
        st.subheader("Generation")
        temperature = st.slider("Temperature", 0.0, 1.5, float(settings.temperature), 0.05)
        max_tokens = st.number_input(
            "Max tokens / num_predict",
            min_value=16,
            max_value=4096,
            value=int(settings.max_tokens),
            step=16,
        )
        top_p = st.slider("Top p", 0.1, 1.0, float(settings.top_p), 0.05)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("RAG")
    rag_left, rag_right = st.columns(2, gap="large")
    with rag_left:
        retrieval_k = st.number_input(
            "Initial retrieval k",
            min_value=1,
            max_value=50,
            value=int(settings.retrieval_k),
            step=1,
        )
        final_retrieval_k = st.number_input(
            "Final retrieval k",
            min_value=1,
            max_value=20,
            value=int(settings.final_retrieval_k),
            step=1,
        )
    with rag_right:
        rerank_enabled = st.toggle("Rerank enabled", value=bool(settings.rerank_enabled))
        general_fallback_enabled = st.toggle(
            "General fallback enabled",
            value=bool(settings.general_fallback_enabled),
        )
    st.markdown("</div>", unsafe_allow_html=True)

    updated_settings = RuntimeSettings(
        provider=provider,
        ollama_base_url=ollama_base_url.strip() or "http://localhost:11434",
        ollama_model=ollama_model.strip() or settings.ollama_model,
        openai_api_key=openai_api_key.strip(),
        openai_model=openai_model.strip() or settings.openai_model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        top_p=float(top_p),
        retrieval_k=int(retrieval_k),
        final_retrieval_k=int(final_retrieval_k),
        rerank_enabled=bool(rerank_enabled),
        general_fallback_enabled=bool(general_fallback_enabled),
    )
    st.session_state["runtime_settings"] = updated_settings

    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("Save Settings", use_container_width=True):
            persist_runtime_settings(updated_settings)
            st.success("Saved non-secret settings locally.")
    with reset_col:
        if st.button("Reset Session", use_container_width=True):
            st.session_state["runtime_settings"] = load_runtime_settings()
            st.rerun()

    st.caption("Changes apply immediately to new answers on the chat page. The OpenAI API key stays session-only.")


def main() -> None:
    """Run the dedicated Settings page."""

    apply_page_styles()
    initialize_session_state()
    render_settings_page()


if __name__ == "__main__":
    main()
