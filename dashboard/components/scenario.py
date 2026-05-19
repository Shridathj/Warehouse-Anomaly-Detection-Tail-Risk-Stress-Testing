from __future__ import annotations

import streamlit as st

from dashboard.artifacts import artifacts_exist, save_artifacts
from dashboard.cache import load_cached_artifacts, load_scenario_dataframe
from dashboard.components.stage import render_stage
from dashboard.pipeline import execute_scenario
from dashboard.runtime import is_cloud_host, render_compute_banner
from src.config import SCENARIO_CONFIGS

MODE_BUNDLED = "bundled"
MODE_LIVE = "live"

SCENARIO_DESCRIPTIONS = {
    1: (
        "Positive-quantity gross orders only (no refunds). "
        "Represents maximum tail-risk exposure under firm demand."
    ),
    2: (
        "Netted quantities after cancellations and partial refunds. "
        "Reflects realised demand after operational reversals."
    ),
}


def _cache_key(scenario_id: int) -> str:
    return f"scenario_{scenario_id}_outputs"


def _mode_key(scenario_id: int) -> str:
    return f"scenario_{scenario_id}_mode"


def render_scenario(scenario_id: int) -> None:
    cfg = SCENARIO_CONFIGS[scenario_id]
    cache_key = _cache_key(scenario_id)
    run_flag = f"run_requested_{scenario_id}"
    mode_key = _mode_key(scenario_id)
    has_bundle = artifacts_exist(scenario_id)

    st.markdown(
        f"""
        <div class="scenario-banner">
          <h2>{cfg['label']}</h2>
          <p>{SCENARIO_DESCRIPTIONS[scenario_id]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_compute_banner()

    if mode_key not in st.session_state:
        st.session_state[mode_key] = MODE_BUNDLED if has_bundle else MODE_LIVE

    mode = st.session_state[mode_key]
    cols = st.columns(2)
    with cols[0]:
        st.caption("Results source")
        st.markdown(
            "**Bundled (instant)**" if mode == MODE_BUNDLED else "**Live pipeline**"
        )
    with cols[1]:
        if has_bundle:
            st.caption("Tip")
            st.markdown("Use sidebar **Load bundled** for fastest view.")

    if not has_bundle and mode == MODE_BUNDLED:
        st.session_state[mode_key] = MODE_LIVE
        mode = MODE_LIVE
        st.info(
            "No bundled cache in `results/dashboard_cache/`. "
            "Run the live pipeline once (sidebar), then results are saved for next time."
        )

    should_run_live = st.session_state.pop(run_flag, False)

    if (
        mode == MODE_BUNDLED
        and has_bundle
        and cache_key not in st.session_state
        and not should_run_live
    ):
        bundled = load_cached_artifacts(scenario_id)
        if bundled:
            st.session_state[cache_key] = bundled

    if should_run_live and mode == MODE_LIVE:
        progress = st.progress(0.0, text="Initialising pipeline…")
        status = st.empty()
        host = "cloud server" if is_cloud_host() else "this machine"
        loader_key = "gross" if scenario_id == 1 else "netted"

        def on_progress(step: int, total: int, title: str) -> None:
            progress.progress(step / total, text=f"{title} ({step}/{total})")
            status.caption(f"Running on **{host}**: {title}")

        try:
            df = load_scenario_dataframe(loader_key)
            with st.spinner(f"Running pipeline on {host}…"):
                outputs = execute_scenario(
                    scenario_id,
                    df=df,
                    progress_callback=on_progress,
                    persist_plots=True,
                )
            st.session_state[cache_key] = outputs
            save_artifacts(scenario_id, outputs)
            progress.progress(1.0, text="Complete")
            status.success(f"Pipeline finished on {host}. Results cached for faster reload.")
        except Exception as exc:
            status.error(f"Pipeline failed: {exc}")
            return
        finally:
            progress.empty()
            status.empty()

    outputs = st.session_state.get(cache_key)
    if not outputs:
        st.warning(
            "No results yet. Use the sidebar: **Load bundled results** (fastest) or "
            "**Run live pipeline**."
        )
        return

    source = "bundled cache" if mode == MODE_BUNDLED else "live run"
    st.markdown(f"**{len(outputs)} stages** · {source}")
    st.divider()

    for index, stage in enumerate(outputs, start=1):
        render_stage(stage, index=index)
        if index < len(outputs):
            st.divider()
