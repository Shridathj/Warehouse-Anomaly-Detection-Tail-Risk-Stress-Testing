"""Streamlit-cached loaders for fast, repeatable dashboard sessions."""

from __future__ import annotations

import streamlit as st

from dashboard.artifacts import artifacts_exist, load_artifacts
from src.data.loader import load_and_clean_uci


@st.cache_data(show_spinner="Loading UCI retail data…", ttl=3600)
def load_scenario_dataframe(scenario_key: str):
    """Cache cleaned data per scenario (gross / netted)."""
    return load_and_clean_uci(scenario=scenario_key)


@st.cache_data(show_spinner="Loading saved dashboard results…")
def load_cached_artifacts(scenario_id: int):
    """Cache deserialized bundled results for instant UI."""
    if not artifacts_exist(scenario_id):
        return None
    return load_artifacts(scenario_id)
