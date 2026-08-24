"""Single source of truth for project filesystem locations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        marker = parent / "streamlit_dashboard.py"
        src = parent / "src"
        if marker.exists() and src.is_dir():
            return parent
        if (parent / "pyproject.toml").exists() and src.is_dir():
            return parent
    return current.parents[1]


def relative_to_root(path: str | Path, root: Path | None = None) -> str:
    candidate = Path(path)
    base = Path(root) if root is not None else project_root()
    try:
        return candidate.resolve().relative_to(base.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


class ProjectPaths:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else project_root()

    @property
    def static(self) -> Path:
        return self.root / "static"

    @property
    def project_report(self) -> Path:
        return self.root / "project_report"

    @property
    def dashboard_cache(self) -> Path:
        return self.root / "results" / "dashboard_cache"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    def scenario_bundle(self, scenario_id: int) -> Path:
        return self.dashboard_cache / f"scenario_{int(scenario_id)}" / "stages.pkl"

    def scenario_plot_dirs(self, scenario_id: int) -> list[Path]:
        sid = int(scenario_id)
        return [
            self.plots / f"scenario{sid}",
            self.results / f"scenario{sid}" / "plots",
        ]


PATHS = ProjectPaths()
