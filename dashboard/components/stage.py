from __future__ import annotations

import re

import matplotlib.pyplot as plt
import streamlit as st

from dashboard.pipeline import StageOutput

_NUMBERED = re.compile(r"^\s*\d+\)\s*")


def render_stage(stage: StageOutput, index: int) -> None:
    has_logs = bool(stage.logs)
    has_plots = bool(stage.mpl_figs or stage.plotly_figs or stage.mpl_pngs)

    with st.container(border=True):
        heading = _NUMBERED.sub("", stage.title).strip() or stage.title
        st.markdown(
            f"""
            <div class="stage-head">
              <span class="stage-num">{index:02d}</span>
              <h3 style="margin:0;">{heading}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not has_logs and not has_plots:
            st.info("No captured output or figures for this stage.")
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
        st.plotly_chart(fig, width="stretch")
    for png in stage.mpl_pngs:
        st.image(png, width="stretch")
    for fig in stage.mpl_figs:
        st.pyplot(fig, clear_figure=True, width="stretch")
    if stage.mpl_figs:
        plt.close("all")
