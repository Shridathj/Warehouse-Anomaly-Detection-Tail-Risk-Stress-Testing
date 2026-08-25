from __future__ import annotations

import streamlit as st

from dashboard.cache import backend_health, load_readme
from dashboard.components.sidebar import (
    PAGE_HOME,
    PAGE_SCENARIO_1,
    PAGE_SCENARIO_2,
    goto_page,
    render_page_pager,
)
from dashboard.theme import chip
from src.config import SCENARIO_CONFIGS


def render_home() -> None:
    health = backend_health()
    art1 = health["artifacts"]["1"]
    art2 = health["artifacts"]["2"]

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">UCI Online Retail · 2010–2011</div>
          <h1>Warehouse Anomaly Detection<br>&amp; Tail-Risk Stress Tester</h1>
          <p>
            Measure preventable loss on extreme high-value (“dragon”) orders when
            warehouse service levels fall from the 99th to the 95th percentile.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="finding">
          <div class="finding-label">Key finding</div>
          <p class="finding-body">
            Holding 99th-percentile SLA on high-value orders can remove nearly all
            preventable tail-risk loss. At a 95th-percentile SLA the directional
            annual exposure is <strong>£77k – £268k</strong>. Figures are
            assumption-driven and not operational estimates.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        _scenario_tile(1, art1, "Scenario 1 — Gross (max risk)")
        st.button(
            "Open Scenario 1",
            width="stretch",
            on_click=goto_page,
            args=(PAGE_SCENARIO_1,),
            key="open_s1",
        )
    with col_b:
        _scenario_tile(2, art2, "Scenario 2 — Netted (refunds)")
        st.button(
            "Open Scenario 2",
            width="stretch",
            on_click=goto_page,
            args=(PAGE_SCENARIO_2,),
            key="open_s2",
        )

    st.markdown("### Pipeline")
    st.markdown(
        """
        <div class="rail">
          <div class="rail-item"><span>01</span>Global stats &amp; EVT</div>
          <div class="rail-item"><span>02</span>Delay simulation</div>
          <div class="rail-item"><span>03</span>VaR / ES</div>
          <div class="rail-item"><span>04</span>Monte Carlo</div>
          <div class="rail-item"><span>05</span>Causal engine</div>
          <div class="rail-item"><span>06</span>Hawkes + Kalman</div>
          <div class="rail-item"><span>07</span>Purged backtest</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("README"):
        st.markdown(load_readme())

    render_page_pager(PAGE_HOME, location="bottom")


def _scenario_tile(scenario_id: int, meta: dict, title: str) -> None:
    cfg = SCENARIO_CONFIGS[scenario_id]
    status = (
        chip("cache ready", "ok") if meta.get("exists") else chip("no cache", "miss")
    )
    size = meta.get("size_label", "—")
    st.markdown(
        f"""
        <div class="tile">
          <div class="tile-kicker">Scenario {scenario_id} · {size}</div>
          <h3>{title}</h3>
          {status}
          <ul>
            <li>Dragon rate: <code>{cfg['DRAGON_PCT']:.5f}</code></li>
            <li>SLA breach: <code>{cfg['SLA_BREACH_MIN']}</code> min</li>
            <li>Target annual dragons: <code>{cfg['TARGET_ANNUAL_DRAGONS']}</code></li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
