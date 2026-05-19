from __future__ import annotations

import streamlit as st

from dashboard.runtime import cloud_app_url, is_cloud_host
from src.config import SCENARIO_CONFIGS


def render_home(readme_text: str) -> None:
    st.markdown(
        """
        <div class="dashboard-hero">
          <h1>Warehouse Anomaly Detection &amp; Tail-Risk Stress Tester</h1>
          <p>
            Quantify preventable financial loss from extreme high-value orders when
            fulfilment service levels degrade. Select a scenario in the sidebar to run
            the full analytics pipeline and review outputs in order.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Scenarios", "2", help="Gross (max risk) vs netted (refunds/cancellations)")
    with c2:
        st.metric("Pipeline stages", "6", help="From global stats through backtest")
    with c3:
        st.metric("Data source", "UCI Online Retail", help="Loaded via src/data/loader.py")

    st.markdown("### Scenario comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        cfg1 = SCENARIO_CONFIGS[1]
        st.markdown(f"**{cfg1['label']}**")
        st.markdown(
            f"- Dragon rate: `{cfg1['DRAGON_PCT']:.5f}`\n"
            f"- SLA breach threshold: `{cfg1['SLA_BREACH_MIN']}` min\n"
            f"- Target annual dragons: `{cfg1['TARGET_ANNUAL_DRAGONS']}`"
        )
    with col_b:
        cfg2 = SCENARIO_CONFIGS[2]
        st.markdown(f"**{cfg2['label']}**")
        st.markdown(
            f"- Dragon rate: `{cfg2['DRAGON_PCT']:.5f}`\n"
            f"- SLA breach threshold: `{cfg2['SLA_BREACH_MIN']}` min\n"
            f"- Target annual dragons: `{cfg2['TARGET_ANNUAL_DRAGONS']}`"
        )

    st.markdown("### Project documentation")
    with st.expander("Full README", expanded=False):
        st.markdown(readme_text)

    if is_cloud_host():
        st.info(
            "Running on **Streamlit Cloud**. Open a scenario → **Load bundled results** (fastest) "
            "or **Run live pipeline** once; results are cached for the next visit."
        )
    else:
        url = cloud_app_url()
        if url:
            st.info(f"Prefer cloud compute: [{url}]({url}) · Locally use **Load bundled results** when available.")
        else:
            st.info(
                "Sidebar: **Load bundled results** for instant view, or **Run live pipeline** to recompute. "
                "See `DEPLOYMENT.md` for cloud deployment."
            )
