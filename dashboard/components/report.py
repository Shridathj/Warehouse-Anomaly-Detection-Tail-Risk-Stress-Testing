from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from dashboard.cache import document_signature, load_document_bytes
from dashboard.components.sidebar import PAGE_MONOGRAPH, PAGE_REPORT, render_page_pager
from dashboard.store import STORE
from dashboard.theme import chip


def render_report(doc_id: str) -> None:
    page = PAGE_REPORT if doc_id == "anomaly_summary" else PAGE_MONOGRAPH
    doc = STORE.resolve_document(doc_id)

    st.markdown(
        f"""
        <div class="doc-banner">
          <h2>{doc.title}</h2>
          <p>{doc.description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status = chip("on disk", "ok") if doc.exists else chip("missing", "miss")
    st.markdown(
        f"{status} · `{doc.display_path or doc.filename}` · {doc.size_label}",
        unsafe_allow_html=True,
    )

    if not doc.exists:
        st.error("PDF not found.")
        for path in doc.searched:
            st.code(path, language=None)
        render_page_pager(page, location="bottom")
        return

    payload = load_document_bytes(doc.id, document_signature(doc.id))
    if payload is None:
        payload = Path(doc.path).read_bytes()

    st.download_button(
        label="Download PDF",
        data=payload,
        file_name=doc.filename,
        mime="application/pdf",
        width="stretch",
        key=f"dl_{doc.id}",
    )

    pdf_viewer(
        doc.path,
        width="100%",
        height=860,
        render_text=True,
        key=f"pdf_{doc.id}",
    )

    render_page_pager(page, location="bottom")
