from __future__ import annotations

import streamlit as st

from dashboard.components.home import render_home
from dashboard.components.report import render_report
from dashboard.components.scenario import render_scenario
from dashboard.components.sidebar import (
    PAGE_HOME,
    PAGE_MONOGRAPH,
    PAGE_REPORT,
    PAGE_SCENARIO_1,
    PAGE_SCENARIO_2,
    render_sidebar,
)
from dashboard.theme import inject_custom_css


def main() -> None:
    st.set_page_config(
        page_title="Warehouse Risk Lab",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": (
                "Warehouse Anomaly Detection & Tail-Risk Stress Tester. "
                "Illustrative research framework — not operational advice."
            ),
        },
    )
    inject_custom_css()

    page = render_sidebar()

    if page == PAGE_HOME:
        render_home()
    elif page == PAGE_SCENARIO_1:
        render_scenario(1)
    elif page == PAGE_SCENARIO_2:
        render_scenario(2)
    elif page == PAGE_REPORT:
        render_report("anomaly_summary")
    elif page == PAGE_MONOGRAPH:
        render_report("evt_monograph")


if __name__ == "__main__":
    main()
