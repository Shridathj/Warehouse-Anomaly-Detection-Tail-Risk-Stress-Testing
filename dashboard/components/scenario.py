from __future__ import annotations

from datetime import datetime

import streamlit as st

from dashboard.cache import (
    artifact_signature,
    invalidate_backend_cache,
    load_cached_artifacts,
    load_scenario_dataframe,
)
from dashboard.components.sidebar import PAGE_SCENARIO_1, PAGE_SCENARIO_2, render_page_pager
from dashboard.components.stage import render_stage
from dashboard.metrics import metrics_from_stages
from dashboard.pipeline import execute_scenario
from dashboard.runtime import is_cloud_host
from dashboard.store import STORE, scenario_loader_key
from dashboard.theme import chip
from src.config import SCENARIO_CONFIGS

MODE_BUNDLED = "bundled"
MODE_LIVE = "live"

SCENARIO_DESCRIPTIONS = {
    1: "Positive-quantity orders only. Refunds and cancellations are excluded.",
    2: "Quantities after refunds, cancellations, and partial cancellations.",
}


def _cache_key(scenario_id: int) -> str:
    return f"scenario_{scenario_id}_outputs"


def _mode_key(scenario_id: int) -> str:
    return f"scenario_{scenario_id}_mode"


def _fmt_modified(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y %H:%M")
    except ValueError:
        return iso


def render_scenario(scenario_id: int) -> None:
    cfg = SCENARIO_CONFIGS[scenario_id]
    cache_key = _cache_key(scenario_id)
    run_flag = f"run_requested_{scenario_id}"
    mode_key = _mode_key(scenario_id)
    meta = STORE.artifact_meta(scenario_id)
    has_bundle = meta.exists

    page = PAGE_SCENARIO_1 if scenario_id == 1 else PAGE_SCENARIO_2

    st.markdown(
        f"""
        <div class="scenario-banner">
          <h2>{cfg['label']}</h2>
          <p>{SCENARIO_DESCRIPTIONS[scenario_id]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode_key not in st.session_state:
        st.session_state[mode_key] = MODE_BUNDLED if has_bundle else MODE_LIVE

    mode = st.session_state[mode_key]
    if not has_bundle and mode == MODE_BUNDLED:
        st.session_state[mode_key] = MODE_LIVE
        mode = MODE_LIVE
        st.info(
            "No cache in `results/dashboard_cache/`. "
            "Run the live pipeline from the sidebar; results are saved for next time."
        )

    source_chip = (
        chip("bundled cache", "ok") if mode == MODE_BUNDLED else chip("live run", "warn")
    )
    st.markdown(
        f"{source_chip} · `{meta.display_path}` · {meta.size_label} · {_fmt_modified(meta.modified_iso)}",
        unsafe_allow_html=True,
    )

    should_run_live = st.session_state.pop(run_flag, False)

    if (
        mode == MODE_BUNDLED
        and has_bundle
        and cache_key not in st.session_state
        and not should_run_live
    ):
        result = load_cached_artifacts(scenario_id, artifact_signature(scenario_id))
        if result.stages:
            st.session_state[cache_key] = result.stages
            if result.fallback_used:
                st.warning(
                    result.error or "Cache unreadable; showing PNG charts from disk."
                )
        else:
            st.error(result.error or "Could not load bundled results.")

    if should_run_live and mode == MODE_LIVE:
        progress = st.progress(0.0, text="Starting pipeline…")
        status = st.empty()
        host = "cloud server" if is_cloud_host() else "this machine"
        loader_key = scenario_loader_key(scenario_id)

        def on_progress(step: int, total: int, title: str) -> None:
            progress.progress(step / total, text=f"{title} ({step}/{total})")
            status.caption(f"Running on **{host}**: {title}")

        try:
            df = load_scenario_dataframe(loader_key)
            outputs = execute_scenario(
                scenario_id,
                df=df,
                progress_callback=on_progress,
                persist_plots=True,
            )
            st.session_state[cache_key] = outputs
            try:
                STORE.save_stages(scenario_id, outputs)
                invalidate_backend_cache()
            except Exception as save_exc:
                st.warning(f"Results are in this session, but the cache was not written: {save_exc}")
            progress.progress(1.0, text="Complete")
            status.success(f"Pipeline finished on {host}.")
        except Exception as exc:
            progress.empty()
            status.error(f"Pipeline failed: {exc}")
            if cache_key not in st.session_state:
                return
            st.info("Showing the last results still in this session.")
        else:
            progress.empty()

    outputs = st.session_state.get(cache_key)
    if not outputs:
        st.warning(
            "No results yet. Use **Load bundled results** or **Run live pipeline** in the sidebar."
        )
        render_page_pager(page, location="bottom")
        return

    headlines = metrics_from_stages(outputs)
    if headlines.any():
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Transactions", headlines.transactions or "—")
        with k2:
            st.metric("Dragon events", headlines.dragons or "—")
        with k3:
            st.metric("ES 95%", f"£{headlines.es95}" if headlines.es95 else "—")
        with k4:
            if headlines.unfulfilled:
                st.metric("Unfulfilled dragons", headlines.unfulfilled)
            elif headlines.hill_xi:
                st.metric("Hill ξ", headlines.hill_xi)
            else:
                st.metric("VaR 95%", f"£{headlines.var95}" if headlines.var95 else "—")

    st.caption(f"{len(outputs)} stages · {mode}")
    for index, stage in enumerate(outputs, start=1):
        render_stage(stage, index=index)
        if index < len(outputs):
            st.divider()

    render_page_pager(page, location="bottom")
