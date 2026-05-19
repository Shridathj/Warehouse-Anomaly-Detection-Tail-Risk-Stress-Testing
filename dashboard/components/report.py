from __future__ import annotations

from pathlib import Path

import streamlit as st

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
_PDF_FILENAME = "updated_anomaly_summary.pdf"
_PDF_SRC = f"app/static/{_PDF_FILENAME}"


def render_report() -> None:
    st.markdown(
        """
        <div class="scenario-banner">
          <h2>Project Report</h2>
          <p>Updated anomaly summary — full findings and methodology.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_disk_path = _STATIC_DIR / _PDF_FILENAME

    if not pdf_disk_path.exists():
        st.error(
            f"`static/{_PDF_FILENAME}` not found. "
            "Copy `project_report/updated_anomaly_summary.pdf` → `static/updated_anomaly_summary.pdf` "
            "and restart the app."
        )
        return

    st.markdown(
        f"""
        <iframe
            src="{_PDF_SRC}"
            width="100%"
            height="900px"
            style="border:none; border-radius:6px;"
        ></iframe>
        """,
        unsafe_allow_html=True,
    )