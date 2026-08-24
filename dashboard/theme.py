from __future__ import annotations

import streamlit as st

_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,680&family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;500;600;700&display=swap');

  :root {
    --paper: #f3eee4;
    --paper-2: #ebe4d6;
    --sheet: #fffdf8;
    --ink: #14171c;
    --ink-soft: #2c323c;
    --muted: #6d675c;
    --line: #d8d0c0;
    --gold: #b0893e;
    --gold-soft: #e7d7b1;
    --crimson: #8f2d2d;
    --teal: #1f5c57;
    --ok: #2f6f4f;
    --sidebar: #141820;
    --sidebar-text: #efe8da;
    --sidebar-muted: #a79c88;
  }

  .stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
  }

  .block-container {
    padding-top: 1.35rem;
    padding-bottom: 3.2rem;
    max-width: 1220px;
  }

  h1, h2, h3, h4, .hero h1, .doc-banner h2, .scenario-banner h2, .sidebar-brand {
    font-family: Fraunces, Georgia, serif;
    letter-spacing: -0.02em;
    color: var(--ink) !important;
  }

  p, label, .stMarkdown, .stCaption, span, li {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
  }

  code, pre, kbd, .stCode, div[data-testid="stCode"] {
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  }

  #MainMenu, footer, div[data-testid="stDecoration"],
  .stDeployButton, [data-testid="stMainMenu"],
  [data-testid="stToolbarActions"] {
    display: none !important;
    visibility: hidden;
  }

  header[data-testid="stHeader"] {
    background: transparent !important;
    display: flex !important;
    visibility: visible !important;
    pointer-events: none;
  }
  header[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
  }
  header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"],
  [data-testid="stExpandSidebarButton"] {
    pointer-events: auto;
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999;
  }
  [data-testid="stExpandSidebarButton"] {
    background: var(--sidebar) !important;
    color: var(--sidebar-text) !important;
    border: 1px solid #3a414e !important;
    border-radius: 8px !important;
  }
  [data-testid="stSidebarCollapseButton"] {
    display: inline !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  section[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid #000;
  }
  section[data-testid="stSidebar"] * {
    color: var(--sidebar-text) !important;
  }
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] .sidebar-tagline {
    color: var(--sidebar-muted) !important;
  }
  section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
  }

  .brand-row {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    margin-bottom: 1.15rem;
  }
  .brand-mark {
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 8px;
    background: linear-gradient(160deg, #d4b06a 0%, #8a6a2c 100%);
    color: #141820;
    font-family: Fraunces, Georgia, serif !important;
    font-style: normal;
    font-weight: 680;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .sidebar-brand {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 680;
    color: var(--sidebar-text) !important;
    line-height: 1.15;
  }
  .sidebar-tagline {
    margin: 0.15rem 0 0 0;
    font-size: 0.76rem;
    color: var(--sidebar-muted) !important;
    line-height: 1.35;
  }

  section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.42rem 0.65rem;
    margin-bottom: 0.18rem;
  }
  section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: rgba(255,253,248,0.06);
  }
  section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(176, 137, 62, 0.16);
    border-color: rgba(176, 137, 62, 0.45);
  }

  section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: var(--sidebar-text) !important;
    border: 1px solid #3a414e;
    border-radius: 8px;
    font-weight: 600;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--gold);
    color: #fff !important;
  }
  section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--gold);
    color: #141820 !important;
    border-color: var(--gold);
  }

  .hero {
    background: linear-gradient(145deg, #141820 0%, #1c2430 58%, #243043 100%);
    color: #f6f0e4;
    border-radius: 16px;
    padding: 1.45rem 1.85rem 1.25rem 1.85rem;
    margin-bottom: 1.05rem;
    border: 1px solid #0c1016;
    position: relative;
    overflow: hidden;
  }
  .hero:after {
    content: "";
    position: absolute;
    right: -40px;
    top: -50px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(176,137,62,0.28), transparent 68%);
    pointer-events: none;
  }
  .hero-kicker {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #d4b06a;
    margin-bottom: 0.7rem;
  }
  .hero h1 {
    margin: 0 0 0.55rem 0;
    font-size: 1.85rem;
    font-weight: 680;
    color: #f7f1e6 !important;
    line-height: 1.18;
  }
  .hero p {
    margin: 0;
    color: #d7d0c3;
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 46rem;
  }

  .finding {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-left: 5px solid var(--gold);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.25rem 0 1.0rem 0;
  }
  .finding-label {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.3rem;
  }
  .finding-body {
    color: var(--ink-soft);
    font-size: 0.98rem;
    line-height: 1.5;
    margin: 0;
  }

  .scenario-banner, .doc-banner {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-left: 5px solid var(--crimson);
    border-radius: 12px;
    padding: 1.05rem 1.3rem;
    margin-bottom: 1.2rem;
  }
  .doc-banner { border-left-color: var(--gold); }
  .scenario-banner h2, .doc-banner h2 {
    margin: 0 0 0.35rem 0;
    font-size: 1.45rem;
  }
  .scenario-banner p, .doc-banner p {
    margin: 0;
    color: var(--muted);
    font-size: 0.95rem;
  }

  .tile {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.95rem 1.1rem 0.9rem 1.1rem;
    height: 100%;
  }
  .tile-kicker {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--crimson);
    margin-bottom: 0.35rem;
  }
  .tile h3 {
    margin: 0 0 0.45rem 0;
    font-size: 1.12rem;
  }
  .tile ul {
    margin: 0;
    padding-left: 1.05rem;
    color: var(--ink-soft);
    font-size: 0.92rem;
  }
  .tile code {
    color: var(--teal);
  }

  .rail {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.35rem 0 1.3rem 0;
  }
  .rail-item {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.32rem 0.7rem 0.32rem 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.82rem;
    color: var(--ink-soft);
  }
  .rail-item span {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    background: var(--ink);
    color: var(--gold-soft);
    border-radius: 999px;
    padding: 0.12rem 0.4rem;
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    border-radius: 999px;
    padding: 0.16rem 0.58rem;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .chip-ok { background: #e4f0e6; color: var(--ok); }
  .chip-warn { background: #f7ead0; color: #7a5a18; }
  .chip-miss { background: #f4e1e1; color: var(--crimson); }
  .chip-ink { background: #ece7dc; color: var(--ink); }

  .health-list { margin-top: 0.35rem; }
  .health-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    margin: 0.22rem 0;
    color: var(--sidebar-text) !important;
  }
  .dot {
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot.ok { background: #5dba7a; }
  .dot.miss { background: #d46a6a; }

  .stage-head {
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
    margin-bottom: 0.15rem;
  }
  .stage-num {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.78rem;
    color: var(--gold);
    letter-spacing: 0.08em;
  }

  div[data-testid="stMetric"] {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.85rem 0.95rem;
  }
  div[data-testid="stMetricLabel"] { color: var(--muted); }
  div[data-testid="stMetricValue"] {
    font-family: Fraunces, Georgia, serif;
    color: var(--ink);
  }

  .stButton > button {
    border-radius: 8px;
    border: 1px solid var(--line);
    font-weight: 600;
    background: var(--sheet);
    color: var(--ink);
  }
  .stButton > button[kind="primary"] {
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
  }
  .stButton > button[kind="primary"]:hover {
    background: var(--crimson);
    border-color: var(--crimson);
  }

  div[data-testid="stExpander"] {
    background: var(--sheet);
    border: 1px solid var(--line);
    border-radius: 10px;
  }

  div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.35rem;
  }

  hr { border-color: var(--line) !important; }
</style>
"""


def inject_custom_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def chip(text: str, kind: str = "ink") -> str:
    return f'<span class="chip chip-{kind}">{text}</span>'
