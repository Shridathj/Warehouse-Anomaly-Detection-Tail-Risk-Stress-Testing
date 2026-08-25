"""
Execution of the anomaly detection pipeline.

Run from repository root:
    python run_src.py
"""
import logging
import sys
from pathlib import Path

logging.disable(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from src.config import SCENARIO_CONFIGS
from src.data.loader import load_and_clean_uci
from src.delay_simulation.delays import run_mock_delays
from src.risk.var_es import run_var
from src.risk.monte_carlo import run_monte_carlo
from src.global_statistics.global_stats import (
    run_global_statistics,
    run_evt_gpd,
    run_sku_filter,
    run_param_summary,
)
from src.hwk_kalman_forecasting.mle_kalman import run_backtest as run_hawkes_kalman
from src.causal_engine.causal import run_causal_engine
from src.backtest.backtest import run_quantitative_backtest
from src.utils.plot_paths import plot_output_session, reset_plots_dir, _configure_plotly_export_logging

_configure_plotly_export_logging()


def call(fn, df, **kwargs):
    import inspect

    sig = inspect.signature(fn)
    state = kwargs.get("state", {}) or {}

    if "df" in state:
        df = state["df"]

    if "state" not in sig.parameters:
        kwargs.pop("state", None)

    result = fn(df, **kwargs)
    return result if isinstance(result, dict) else state


def run_scenario_pipeline(scenario: int, df) -> None:
    cfg = SCENARIO_CONFIGS[scenario]
    state = {}
    state = call(run_global_statistics, df, scenario=scenario, state=state)
    state = call(run_evt_gpd, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_sku_filter, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_param_summary, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_mock_delays, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_var, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_monte_carlo, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_causal_engine, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_hawkes_kalman, df, scenario=scenario, cfg=cfg, state=state)
    state = call(run_quantitative_backtest, df, scenario=scenario, cfg=cfg, state=state)


if __name__ == "__main__":
    plots_root = reset_plots_dir()
    print(f"\nCleared and prepared plot output directory: {plots_root.resolve()}")

    print("\n ANOMALY DETECTION PIPELINE - SCENARIO 1 (GROSS)")
    df1 = load_and_clean_uci(scenario="gross")
    with plot_output_session(1):
        run_scenario_pipeline(1, df1)

    print("\n ANOMALY DETECTION PIPELINE - SCENARIO 2 (NETTED)")
    df2 = load_and_clean_uci(scenario="netted")
    with plot_output_session(2):
        run_scenario_pipeline(2, df2)

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Plots saved under: {plots_root.resolve()}")
