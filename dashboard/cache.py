"""Streamlit-cached loaders over the dashboard store."""

from __future__ import annotations

import streamlit as st

from dashboard.store import STORE, StageLoadResult, normalize_loader_key


@st.cache_data(show_spinner="Loading UCI retail data…", ttl=3600)
def load_scenario_dataframe(scenario_key: str):
    """Cache cleaned data per scenario (gross / netted)."""
    from src.data.loader import load_and_clean_uci

    return load_and_clean_uci(scenario=normalize_loader_key(scenario_key))


@st.cache_data(show_spinner="Loading saved dashboard results…")
def load_cached_artifacts(scenario_id: int, signature: str = "") -> StageLoadResult:
    """Cache deserialized bundled results. `signature` busts the cache on file change."""
    del signature
    return STORE.load_stages(int(scenario_id))


@st.cache_data(show_spinner=False)
def load_document_bytes(doc_id: str, signature: str = "") -> bytes | None:
    del signature
    return STORE.read_document_bytes(doc_id)


@st.cache_data(show_spinner=False)
def backend_health() -> dict:
    return STORE.health_snapshot()


@st.cache_data(show_spinner=False)
def load_readme() -> str:
    return STORE.readme_text()


def artifact_signature(scenario_id: int) -> str:
    meta = STORE.artifact_meta(int(scenario_id))
    return f"{meta.exists}:{meta.size_bytes}:{meta.modified_iso}"


def document_signature(doc_id: str) -> str:
    doc = STORE.resolve_document(doc_id)
    return f"{doc.exists}:{doc.size_bytes}:{doc.path}"


def invalidate_backend_cache() -> None:
    load_cached_artifacts.clear()
    load_document_bytes.clear()
    backend_health.clear()
    load_readme.clear()
