"""
Central plot output for run_src.py and Streamlit live pipeline runs.

All saved figures go under:
    plots/scenario1/
    plots/scenario2/
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

PLOTS_ROOT = Path(__file__).resolve().parents[2] / "plots"

_enabled: bool = False
_current_scenario: int | None = None
_auto_mpl_counter: int = 0
_orig_plt_show = None


def scenario_dir(scenario: int) -> Path:
    return PLOTS_ROOT / f"scenario{scenario}"


def reset_plots_dir() -> Path:
    """Delete the entire plots/ tree and recreate scenario subfolders."""
    if PLOTS_ROOT.exists():
        shutil.rmtree(PLOTS_ROOT)
    scenario_dir(1).mkdir(parents=True, exist_ok=True)
    scenario_dir(2).mkdir(parents=True, exist_ok=True)
    return PLOTS_ROOT


def clear_scenario_plots(scenario: int) -> Path:
    """Remove and recreate plots for one scenario (Streamlit per-scenario runs)."""
    target = scenario_dir(scenario)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def is_plot_save_enabled() -> bool:
    return _enabled


def resolve_plot_path(filename: str, scenario: int | None = None) -> Path:
    sid = scenario if scenario is not None else _current_scenario
    if sid is None:
        raise RuntimeError("Plot scenario not set; use plot_output_session() or pass scenario=.")
    name = Path(filename).name
    return scenario_dir(sid) / name


def save_matplotlib_figure(
    filename: str,
    *,
    scenario: int | None = None,
    fig=None,
    **savefig_kwargs,
) -> Path | None:
    if not _enabled:
        return None
    path = resolve_plot_path(filename, scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = fig if fig is not None else plt.gcf()
    defaults = {"dpi": 300, "bbox_inches": "tight"}
    defaults.update(savefig_kwargs)
    target.savefig(path, **defaults)
    return path


def save_matplotlib_figure_auto(fig=None) -> Path | None:
    global _auto_mpl_counter
    _auto_mpl_counter += 1
    return save_matplotlib_figure(
        f"mpl_auto_{_auto_mpl_counter:03d}.png",
        fig=fig,
    )


def save_plotly_figure(
    fig,
    filename: str,
    *,
    scenario: int | None = None,
    width: int = 1200,
    height: int = 680,
    scale: int = 2,
) -> Path | None:
    if not _enabled:
        return None
    path = resolve_plot_path(filename, scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), width=width, height=height, scale=scale)
    except Exception:
        html_path = path.with_suffix(".html")
        fig.write_html(str(html_path))
        return html_path
    return path


def _patched_plt_show(*_args, **_kwargs) -> None:
    """Close figures after display (named saves use save_matplotlib_figure)."""
    plt.close("all")


@contextmanager
def plot_output_session(scenario: int, *, reset_scenario_dir: bool = False):
    """
    Enable disk plot capture for the active scenario.
    Intercepts plt.show() so figures are written under plots/scenario{N}/.
    """
    global _enabled, _current_scenario, _auto_mpl_counter, _orig_plt_show

    if reset_scenario_dir:
        clear_scenario_plots(scenario)

    _auto_mpl_counter = 0
    _current_scenario = scenario
    _enabled = True
    _orig_plt_show = plt.show
    plt.show = _patched_plt_show
    try:
        yield
    finally:
        plt.show = _orig_plt_show
        _enabled = False
        _current_scenario = None
