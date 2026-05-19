"""Download UCI Online Retail dataset when missing (local or cloud)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

UCI_ONLINE_RETAIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)
DEFAULT_FILENAME = "Online Retail.xlsx"


def ensure_online_retail_xlsx(base_dir: Path | None = None) -> Path:
    """
    Ensure ``Online Retail.xlsx`` exists under ``dataset/raw/``.
    Downloads from UCI if absent.
    """
    root = base_dir or Path.cwd()
    dest = root / "dataset" / "raw" / DEFAULT_FILENAME
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DEFAULT_FILENAME} from UCI (~23 MB)…")
    urllib.request.urlretrieve(UCI_ONLINE_RETAIL_URL, dest)
    print(f"Saved to {dest}")
    return dest
