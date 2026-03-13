from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve import get_corpus_state
from src.settings import RuntimeSettings, load_runtime_settings


def apply_page_styles() -> None:
    """Apply lightweight styling shared across Streamlit pages."""

    st.markdown(
        """
        <style>
        :root {
            --page-bg: #0f172a;
            --sidebar-bg: #0b1220;
            --panel-bg: #111827;
            --panel-strong: #172033;
            --border-soft: #1f2937;
            --border-strong: #273449;
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
            --accent: #6366f1;
            --accent-soft: rgba(99, 102, 241, 0.14);
            --shadow-soft: 0 10px 25px rgba(0, 0, 0, 0.25);
            --shadow-strong: 0 18px 40px rgba(0, 0, 0, 0.34);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 22%),
                linear-gradient(180deg, #0b1120 0%, var(--page-bg) 100%);
            color: var(--text-main);
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.8rem;
            max-width: 1120px;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu, footer {
            visibility: hidden;
        }
        section[data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border-soft);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            max-width: 100%;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-main);
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: inherit;
        }
        p, .stMarkdown, .stCaption {
            color: var(--text-main);
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--text-muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--text-main);
        }
        div[data-testid="stChatMessage"] {
            border-radius: 22px;
            border: 1px solid var(--border-soft);
            background: var(--panel-bg);
            padding: 0.2rem 0.35rem;
            box-shadow: var(--shadow-soft);
            max-width: 840px;
            margin-left: auto;
            margin-right: auto;
        }
        .answer-source-badge {
            display: inline-block;
            margin-bottom: 0.7rem;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(99, 102, 241, 0.20);
            background: rgba(99, 102, 241, 0.14);
            color: #c7d2fe;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .hero-shell {
            max-width: 760px;
            margin: 4.5rem auto 2rem auto;
            padding: 2.6rem 2.3rem 2.1rem 2.3rem;
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(17, 24, 39, 0.92) 100%);
            border: 1px solid var(--border-soft);
            box-shadow: var(--shadow-strong);
            text-align: center;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: #c7d2fe;
            font-weight: 600;
            font-size: 0.82rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.7rem;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin: 0 0 0.85rem 0;
            color: var(--text-main);
            font-weight: 700;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.65;
            color: var(--text-muted);
            max-width: 640px;
            margin: 0 auto 1.6rem auto;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.45rem 0 1.15rem 0;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.78rem;
            border-radius: 999px;
            background: var(--panel-bg);
            border: 1px solid var(--border-soft);
            color: var(--text-main);
            font-size: 0.88rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
        }
        .status-chip strong {
            font-weight: 700;
        }
        .section-card {
            background: var(--panel-bg);
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: var(--shadow-soft);
        }
        .source-card {
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            background: var(--panel-strong);
            padding: 0.95rem 1rem;
            margin-bottom: 0.75rem;
        }
        .source-meta {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-bottom: 0.4rem;
        }
        .sidebar-section-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin: 1rem 0 0.65rem 0;
            font-weight: 700;
        }
        .sidebar-history-item {
            padding: 0.65rem 0.75rem;
            border-radius: 14px;
            background: var(--panel-bg);
            border: 1px solid var(--border-soft);
            color: var(--text-main);
            font-size: 0.9rem;
            margin-bottom: 0.55rem;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stChatInput"] {
            background: var(--panel-bg);
            border: 1px solid var(--border-strong);
            border-radius: 30px;
            padding: 0.35rem 0.5rem;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stChatInput"] textarea {
            font-size: 1rem;
            color: var(--text-main) !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-muted) !important;
        }
        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
            border-radius: 999px !important;
            border: none !important;
            color: white !important;
        }
        button[kind="primary"] {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.28);
        }
        button[kind="secondary"] {
            border-radius: 20px !important;
            background: #1f2937 !important;
            border: 1px solid #273449 !important;
            color: var(--text-main) !important;
        }
        button[kind="secondary"]:hover {
            border-color: #374151 !important;
            background: #243042 !important;
            color: white !important;
        }
        .stButton > button {
            transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
        }
        .stFileUploader, div[data-testid="stExpander"], div[data-testid="stAlert"] {
            background: var(--panel-bg);
            border: 1px solid var(--border-soft);
            border-radius: 20px;
        }
        div[data-testid="stExpander"] details summary {
            color: var(--text-main);
        }
        input, textarea, [data-baseweb="input"] > div, [data-baseweb="select"] > div {
            background: var(--panel-bg) !important;
            color: var(--text-main) !important;
            border-color: var(--border-soft) !important;
        }
        [data-baseweb="tag"] {
            background: var(--panel-strong) !important;
            color: var(--text-main) !important;
        }
        a {
            color: #a5b4fc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Ensure shared session state exists across Streamlit pages."""

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "runtime_settings" not in st.session_state:
        st.session_state["runtime_settings"] = load_runtime_settings()


def get_runtime_settings() -> RuntimeSettings:
    """Return the active session-scoped runtime settings."""

    initialize_session_state()
    return st.session_state["runtime_settings"]


def get_index_status() -> dict[str, int | bool]:
    """Return shared index status for chat and workspace pages."""

    corpus_state = get_corpus_state()
    return {
        "is_ready": corpus_state.has_indexed_corpus,
        "chunks_indexed": corpus_state.chunks_indexed,
        "documents_count": corpus_state.documents_count,
    }
