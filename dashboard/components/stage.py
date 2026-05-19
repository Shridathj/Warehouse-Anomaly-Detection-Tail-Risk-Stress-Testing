from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from dashboard.pipeline import StageOutput


def render_stage(stage: StageOutput, index: int) -> None:
    has_logs = bool(stage.logs)
    has_plots = bool(stage.mpl_figs or stage.plotly_figs or stage.mpl_pngs)

    with st.container(border=True):
        st.markdown(f"#### {stage.title}")

        if not has_logs and not has_plots:
            st.info("Stage completed with no captured console output or figures.")
            return

        if has_logs and has_plots:
            tab_logs, tab_charts = st.tabs(["Output", "Charts"])
            with tab_logs:
                st.code(stage.logs, language=None)
            with tab_charts:
                _render_charts(stage)
        elif has_logs:
            with st.expander("Pipeline output", expanded=index == 1):
                st.code(stage.logs, language=None)
        else:
            _render_charts(stage)


def _render_charts(stage: StageOutput) -> None:
    for fig in stage.plotly_figs:
        st.plotly_chart(fig, use_container_width=True)
    for png in stage.mpl_pngs:
        st.image(png, use_container_width=True)
    for fig in stage.mpl_figs:
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    if stage.mpl_figs:
        plt.close("all")
