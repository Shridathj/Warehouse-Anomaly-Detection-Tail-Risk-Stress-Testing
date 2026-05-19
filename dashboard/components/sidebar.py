from __future__ import annotations

import streamlit as st

from dashboard.artifacts import artifacts_exist
from dashboard.runtime import cloud_app_url, compute_location_label, is_cloud_host, is_local_host
from src.config import SCENARIO_CONFIGS

PAGE_HOME = "home"
PAGE_SCENARIO_1 = "scenario_1"
PAGE_SCENARIO_2 = "scenario_2"

NAV_OPTIONS = {
    "Overview": PAGE_HOME,
    "Scenario 1 — Gross (Max Risk)": PAGE_SCENARIO_1,
    "Scenario 2 — Netted": PAGE_SCENARIO_2,
}


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">Warehouse Risk Lab</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-tagline">Tail-risk stress testing &amp; anomaly analytics</p>',
            unsafe_allow_html=True,
        )

        st.caption("Compute")
        st.markdown(f"**{compute_location_label()}**")

        if is_local_host():
            url = cloud_app_url()
            if url:
                st.link_button("Open cloud app (recommended)", url, use_container_width=True)

        selected_label = st.radio(
            "Navigation",
            options=list(NAV_OPTIONS.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )
        page = NAV_OPTIONS[selected_label]

        st.divider()

        if page != PAGE_HOME:
            scenario_id = 1 if page == PAGE_SCENARIO_1 else 2
            cfg = SCENARIO_CONFIGS[scenario_id]
            st.caption("Scenario")
            st.markdown(f"**{cfg['label']}**")

            if artifacts_exist(scenario_id):
                if st.button("Load bundled results", use_container_width=True):
                    st.session_state[f"scenario_{scenario_id}_mode"] = "bundled"
                    st.session_state.pop(f"scenario_{scenario_id}_outputs", None)
                    st.rerun()

            run_label = (
                "Run live pipeline (cloud)"
                if is_cloud_host()
                else "Run live pipeline"
            )
            if st.button(run_label, type="primary", use_container_width=True):
                st.session_state[f"run_requested_{scenario_id}"] = True
                st.session_state[f"scenario_{scenario_id}_mode"] = "live"
                st.rerun()

            if st.button("Clear session results", use_container_width=True):
                st.session_state.pop(f"scenario_{scenario_id}_outputs", None)
                st.session_state.pop(f"run_requested_{scenario_id}", None)
                st.rerun()

        st.divider()
        st.caption("Backend plots (live runs)")
        st.markdown("`plots/scenario1/` · `plots/scenario2/`")

    return page
