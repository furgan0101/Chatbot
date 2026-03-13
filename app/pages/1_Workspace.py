from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_shared import apply_page_styles, get_index_status, initialize_session_state
from src.embed import rebuild_vector_store
from src.ingest import load_documents, save_uploaded_pdfs
from src.utils import DOCS_DIR


st.set_page_config(page_title="Workspace", layout="wide")


def render_workspace_page() -> None:
    """Render file management and indexing controls away from the main chat page."""

    documents = load_documents()
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    index_status = get_index_status()

    st.title("Workspace")
    st.write("Upload course PDFs, review what is available, and rebuild the retrieval index when needed.")

    hero = st.columns([1.2, 1, 1])
    hero[0].metric("Readable PDFs", len(documents))
    hero[1].metric("Indexed Chunks", int(index_status["chunks_indexed"]))
    hero[2].metric("Index Status", "Ready" if index_status["is_ready"] else "Not built")

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Upload PDFs")
        st.write("Add new PDF files here. Saved files go into `data/docs/` and become available for indexing.")
        uploaded_files = st.file_uploader(
            "PDF uploader",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once.",
        )
        if st.button("Save Uploaded PDFs", use_container_width=True):
            if not uploaded_files:
                st.info("Choose one or more PDF files before saving.")
            else:
                saved_files = save_uploaded_pdfs(uploaded_files)
                st.success(f"Saved {len(saved_files)} file(s): {', '.join(saved_files)}")
                st.rerun()
        st.caption(f"Documents folder: {DOCS_DIR}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Available PDFs")
        if not pdf_files:
            st.info("No PDF files were found in `data/docs/`.")
        elif documents:
            for document in documents:
                st.markdown(
                    f"<div class='source-card'><div class='source-meta'>Ready for indexing</div><div>{document.source_name}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("PDF files were found, but none had extractable text.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Index")
        st.write(
            "Rebuild the FAISS index after uploading or replacing files so retrieval reflects the latest workspace."
        )
        if st.button("Build / Rebuild Index", use_container_width=True):
            with st.spinner("Building the vector index from PDFs..."):
                stats = rebuild_vector_store()
            if stats["chunks_indexed"] == 0:
                st.warning("No chunks were indexed. Check that your PDFs contain extractable text.")
            else:
                message = (
                    f"Indexed {stats['documents_indexed']} canonical document(s) into "
                    f"{stats['chunks_indexed']} chunk(s)."
                )
                if stats.get("duplicates_skipped"):
                    message += f" Skipped {stats['duplicates_skipped']} duplicate PDF(s)."
                st.success(message)
                st.rerun()

        if not index_status["is_ready"]:
            st.caption("The chat page will stay disabled until the index is built.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Shortcuts")
        st.write("Use the sidebar navigation to return to `Chat` or adjust models in `Settings`.")
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Run the Workspace page."""

    apply_page_styles()
    initialize_session_state()
    render_workspace_page()


if __name__ == "__main__":
    main()
