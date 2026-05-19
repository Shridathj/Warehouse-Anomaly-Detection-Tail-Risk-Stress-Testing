"""Detect where the dashboard is running (local laptop vs cloud host)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from urllib.parse import urlparse

import streamlit as st

_HTTPS_URL = re.compile(r"^https://", re.IGNORECASE)


@lru_cache(maxsize=1)
def compute_environment() -> str:
    if os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud":
        return "cloud"
    if os.environ.get("IS_STREAMLIT_CLOUD", "").lower() in {"1", "true", "yes"}:
        return "cloud"
    return "local"


def is_cloud_host() -> bool:
    return compute_environment() == "cloud"


def is_local_host() -> bool:
    return not is_cloud_host()


def _sanitize_url(raw: str | None) -> str | None:
    if not raw:
        return None
    url = str(raw).strip().rstrip("/")
    if not _HTTPS_URL.match(url):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return url


def cloud_app_url() -> str | None:
    try:
        url = st.secrets.get("dashboard", {}).get("cloud_url")
        safe = _sanitize_url(url)
        if safe:
            return safe
    except (FileNotFoundError, AttributeError, KeyError, TypeError):
        pass
    return _sanitize_url(os.environ.get("DASHBOARD_CLOUD_URL"))


def compute_location_label() -> str:
    if is_cloud_host():
        return "Cloud (Streamlit server)"
    return "This laptop (local)"


def render_compute_banner() -> None:
    if is_cloud_host():
        st.success("Compute runs on the **cloud server** — this device only shows the UI.")
        return

    url = cloud_app_url()
    if url:
        st.info(
            "For faster runs without loading this laptop, use your cloud deployment: "
            f"[Open cloud app]({url})"
        )
