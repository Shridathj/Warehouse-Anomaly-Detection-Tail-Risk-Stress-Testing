from __future__ import annotations

import streamlit as st

from dashboard.cache import backend_health
from dashboard.runtime import cloud_app_url, compute_location_label, is_cloud_host, is_local_host
from src.config import SCENARIO_CONFIGS

PAGE_HOME = "home"
PAGE_SCENARIO_1 = "scenario_1"
PAGE_SCENARIO_2 = "scenario_2"
PAGE_REPORT = "project_report"
PAGE_MONOGRAPH = "evt_monograph"

NAV_ITEMS: list[tuple[str, str]] = [
    ("Overview", PAGE_HOME),
    ("Scenario 1 — Gross", PAGE_SCENARIO_1),
    ("Scenario 2 — Netted", PAGE_SCENARIO_2),
    ("Anomaly Summary", PAGE_REPORT),
    ("EVT Monograph", PAGE_MONOGRAPH),
]
NAV_OPTIONS = dict(NAV_ITEMS)
LABEL_BY_PAGE = {page: label for label, page in NAV_ITEMS}
PAGE_FLOW = [page for _label, page in NAV_ITEMS]

_QUERY_TO_PAGE = {
    "overview": PAGE_HOME,
    "home": PAGE_HOME,
    "scenario_1": PAGE_SCENARIO_1,
    "scenario_2": PAGE_SCENARIO_2,
    "report": PAGE_REPORT,
    "monograph": PAGE_MONOGRAPH,
}
_PAGE_TO_QUERY = {
    PAGE_HOME: "overview",
    PAGE_SCENARIO_1: "scenario_1",
    PAGE_SCENARIO_2: "scenario_2",
    PAGE_REPORT: "report",
    PAGE_MONOGRAPH: "monograph",
}
_PENDING_NAV = "_pending_nav"


def _page_from_query() -> str | None:
    raw = st.query_params.get("page", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return _QUERY_TO_PAGE.get(str(raw).strip().lower())


def _set_page_query(page: str) -> None:
    desired = _PAGE_TO_QUERY[page]
    if st.query_params.get("page") != desired:
        st.query_params["page"] = desired


def goto_page(page: str) -> None:
    st.session_state[_PENDING_NAV] = page
    _set_page_query(page)


def neighbors(page: str) -> tuple[str | None, str | None]:
    idx = PAGE_FLOW.index(page)
    prev_page = PAGE_FLOW[idx - 1] if idx > 0 else None
    next_page = PAGE_FLOW[idx + 1] if idx < len(PAGE_FLOW) - 1 else None
    return prev_page, next_page


def render_page_pager(page: str, *, location: str) -> None:
    prev_page, next_page = neighbors(page)
    left, right = st.columns(2)
    with left:
        if prev_page:
            st.button(
                "Back",
                width="stretch",
                on_click=goto_page,
                args=(prev_page,),
                key=f"back_{location}_{page}",
                help=LABEL_BY_PAGE[prev_page],
            )
    with right:
        if next_page:
            st.button(
                "Next",
                type="primary",
                width="stretch",
                on_click=goto_page,
                args=(next_page,),
                key=f"next_{location}_{page}",
                help=LABEL_BY_PAGE[next_page],
            )


def _on_nav_radio() -> None:
    label = st.session_state.get("nav_radio")
    page = NAV_OPTIONS.get(label)
    if page:
        _set_page_query(page)


def _apply_pending_nav() -> None:
    pending = st.session_state.pop(_PENDING_NAV, None)
    if pending:
        st.session_state.nav_radio = LABEL_BY_PAGE[pending]
        return
    query_page = _page_from_query()
    if query_page and "nav_radio" not in st.session_state:
        st.session_state.nav_radio = LABEL_BY_PAGE[query_page]


def render_sidebar() -> str:
    health = backend_health()
    dataset = health["dataset"]
    art1 = health["artifacts"]["1"]
    art2 = health["artifacts"]["2"]
    docs = health["documents"]
    docs_ready = sum(1 for doc in docs if doc.get("exists"))

    _apply_pending_nav()

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-row">
              <div class="brand-mark">W</div>
              <div>
                <p class="sidebar-brand">Warehouse Risk Lab</p>
                <p class="sidebar-tagline">Tail-risk stress testing</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Compute")
        st.markdown(f"**{compute_location_label()}**")

        if is_local_host():
            url = cloud_app_url()
            if url:
                st.link_button("Open cloud app", url, width="stretch")

        selected_label = st.radio(
            "Navigation",
            options=[label for label, _page in NAV_ITEMS],
            label_visibility="collapsed",
            key="nav_radio",
            on_change=_on_nav_radio,
        )
        page = NAV_OPTIONS[selected_label]

        st.divider()

        if page in {PAGE_SCENARIO_1, PAGE_SCENARIO_2}:
            scenario_id = 1 if page == PAGE_SCENARIO_1 else 2
            cfg = SCENARIO_CONFIGS[scenario_id]
            st.caption("Scenario")
            st.markdown(f"**{cfg['label']}**")

            bundle_ok = art1["exists"] if scenario_id == 1 else art2["exists"]
            if bundle_ok:
                if st.button("Load bundled results", width="stretch"):
                    st.session_state[f"scenario_{scenario_id}_mode"] = "bundled"
                    st.session_state.pop(f"scenario_{scenario_id}_outputs", None)
                    st.rerun()

            run_label = (
                "Run live pipeline (cloud)"
                if is_cloud_host()
                else "Run live pipeline"
            )
            if st.button(run_label, type="primary", width="stretch"):
                st.session_state[f"run_requested_{scenario_id}"] = True
                st.session_state[f"scenario_{scenario_id}_mode"] = "live"
                st.rerun()

            if st.button("Clear session results", width="stretch"):
                st.session_state.pop(f"scenario_{scenario_id}_outputs", None)
                st.session_state.pop(f"run_requested_{scenario_id}", None)
                st.rerun()

            st.divider()

        ds_class = "ok" if dataset.get("exists") else "miss"
        s1_class = "ok" if art1.get("exists") else "miss"
        s2_class = "ok" if art2.get("exists") else "miss"
        doc_class = "ok" if docs_ready == len(docs) else "miss"
        st.caption("Files")
        st.markdown(
            f"""
            <div class="health-list">
              <div class="health-row"><span class="dot {ds_class}"></span> Dataset · {dataset.get('size_label', '—')}</div>
              <div class="health-row"><span class="dot {s1_class}"></span> Scenario 1 cache · {art1.get('size_label', '—')}</div>
              <div class="health-row"><span class="dot {s2_class}"></span> Scenario 2 cache · {art2.get('size_label', '—')}</div>
              <div class="health-row"><span class="dot {doc_class}"></span> PDFs · {docs_ready}/{len(docs)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page
