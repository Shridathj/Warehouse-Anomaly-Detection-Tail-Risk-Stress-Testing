from __future__ import annotations

import io
import inspect
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from src.backtest.backtest import run_quantitative_backtest
from src.causal_engine.causal import run_causal_engine
from src.config import SCENARIO_CONFIGS
from src.delay_simulation.delays import run_mock_delays
from src.global_statistics.global_stats import (
    run_evt_gpd,
    run_global_statistics,
    run_param_summary,
    run_sku_filter,
)
from src.hwk_kalman_forecasting.mle_kalman import run_backtest as run_hawkes_kalman
from src.risk.monte_carlo import run_monte_carlo
from src.risk.var_es import run_var
from src.utils.plot_paths import plot_output_session


@dataclass
class StageOutput:
    title: str
    logs: str
    mpl_figs: list = field(default_factory=list)
    plotly_figs: list = field(default_factory=list)
    mpl_pngs: list = field(default_factory=list)


class PlotCollector:
    """Collect figures for Streamlit UI only (disk writes happen in src via plot_paths)."""

    def __init__(self) -> None:
        self.plotly_figs: list = []

    def save_plotly(self, fig, filename: str | None = None) -> None:
        del filename  # named saves are handled in src when plot_output_session is active
        self.plotly_figs.append(fig)


@contextmanager
def capture_stage_output():
    """Capture stdout and figures for the dashboard without duplicate disk writes."""
    stdout_buffer = io.StringIO()
    mpl_figs: list = []
    collector = PlotCollector()
    original_show = plt.show

    def _capture_show(*_args, **_kwargs):
        for fig_num in plt.get_fignums():
            mpl_figs.append(plt.figure(fig_num))
        plt.close("all")

    plt.show = _capture_show
    try:
        with redirect_stdout(stdout_buffer):
            yield stdout_buffer, mpl_figs, collector
    finally:
        plt.show = original_show


def call_stage(fn, df, **kwargs):
    sig = inspect.signature(fn)
    state = kwargs.get("state", {}) or {}

    if "df" in state:
        df = state["df"]
    if "state" not in sig.parameters:
        kwargs.pop("state", None)

    result = fn(df, **kwargs)
    return result if isinstance(result, dict) else state


def run_global_bundle(df, scenario: int, cfg: dict, ctx=None, state=None):
    if state is None:
        state = {}
    state = call_stage(run_global_statistics, df, scenario=scenario, cfg=cfg, ctx=ctx, state=state)
    state = call_stage(run_evt_gpd, df, scenario=scenario, cfg=cfg, ctx=ctx, state=state)
    state = call_stage(run_sku_filter, df, scenario=scenario, cfg=cfg, ctx=ctx, state=state)
    state = call_stage(run_param_summary, df, scenario=scenario, cfg=cfg, ctx=ctx, state=state)
    return state


PIPELINE_STAGES = [
    ("1) Global Statistics", run_global_bundle),
    ("2) Delay Simulation", run_mock_delays),
    ("3) Risk (VaR / ES)", run_var),
    ("3) Risk (Monte Carlo)", run_monte_carlo),
    ("4) Causal Engine", run_causal_engine),
    ("5) HWK Kalman Forecasting", run_hawkes_kalman),
    ("6) Backtest", run_quantitative_backtest),
]


def execute_scenario(
    scenario_id: int,
    df=None,
    *,
    progress_callback=None,
    persist_plots: bool = True,
) -> list[StageOutput]:
    """
    Run the full scenario pipeline.

    Parameters
    ----------
    df : optional pre-loaded DataFrame (use cached loader in Streamlit).
    persist_plots : write figures under plots/scenario{N}/ (off for UI-only dev).
    """
    cfg = SCENARIO_CONFIGS[scenario_id]
    if df is None:
        from dashboard.cache import load_scenario_dataframe

        loader_key = "gross" if scenario_id == 1 else "netted"
        df = load_scenario_dataframe(loader_key)

    state: dict = {}
    outputs: list[StageOutput] = []
    total = len(PIPELINE_STAGES)

    plot_ctx = (
        plot_output_session(scenario_id, reset_scenario_dir=persist_plots)
        if persist_plots
        else _null_context()
    )

    with plot_ctx:
        for index, (title, fn) in enumerate(PIPELINE_STAGES, start=1):
            if progress_callback:
                progress_callback(index, total, title)

            with capture_stage_output() as (stdout_buffer, mpl_figs, collector):
                state = call_stage(
                    fn,
                    df,
                    scenario=scenario_id,
                    cfg=cfg,
                    ctx=collector,
                    state=state,
                )

            outputs.append(
                StageOutput(
                    title=title,
                    logs=stdout_buffer.getvalue().strip(),
                    mpl_figs=mpl_figs,
                    plotly_figs=collector.plotly_figs,
                )
            )

    return outputs


@contextmanager
def _null_context():
    yield
