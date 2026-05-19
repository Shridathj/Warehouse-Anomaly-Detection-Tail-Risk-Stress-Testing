from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.components.home import render_home
from dashboard.components.report import render_report                          # ← add
from dashboard.components.scenario import render_scenario
from dashboard.components.sidebar import (
    PAGE_HOME, PAGE_SCENARIO_1, PAGE_SCENARIO_2, PAGE_REPORT,                 # ← add PAGE_REPORT
    render_sidebar,
)
from dashboard.theme import inject_custom_css

_README_PATH = Path(__file__).resolve().parent.parent / "README.md"


@st.cache_data(show_spinner=False)
def _readme_text() -> str:
    return _README_PATH.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(
        page_title="Warehouse Risk Dashboard",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    page = render_sidebar()

    if page == PAGE_HOME:
        render_home(_readme_text())
    elif page == PAGE_SCENARIO_1:
        render_scenario(1)
    elif page == PAGE_SCENARIO_2:
        render_scenario(2)
    elif page == PAGE_REPORT:                                                  # ← add
        render_report()                                                        # ← add


if __name__ == "__main__":
    main()