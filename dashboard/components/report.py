from __future__ import annotations
from pathlib import Path
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

def _find_project_root() -> Path:
    """Locate the project root by walking upward until 'static/' is found."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "static").is_dir():
            return parent
        if (parent / "pyproject.toml").exists() or (parent / "README.md").exists():
            return parent
    return current.parents[2] if len(current.parents) > 2 else current.parent

_PDF_PATH = _find_project_root() / "static" / "updated_anomaly_summary.pdf"

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

    if not _PDF_PATH.exists():
        st.error(f"PDF not found at: `{_PDF_PATH}`")
        st.caption("Please ensure `updated_anomaly_summary.pdf` exists in the `static/` folder at the project root.")
        return

    # Primary viewer – reliable, no download prompt
    pdf_viewer(
        str(_PDF_PATH),
        width=1000,           # adjust as needed (or use height=850)
        render_text=True,     # optional: enables text selection / search
    )

    st.divider()

    # Always keep a clean download option
    with open(_PDF_PATH, "rb") as f:
        st.download_button(
            label="⬇Download full PDF report",
            data=f.read(),
            file_name=_PDF_PATH.name,
            mime="application/pdf",
            use_container_width=True,
        )