"""
Execution of the anomaly detection pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SCENARIO_CONFIGS
from src.data.loader import load_and_clean_uci
from src.delay_simulation.delays import run_mock_delays
from src.risk.var_es import run_var
from src.risk.monte_carlo import run_monte_carlo
from src.global_statistics.global_stats import run_global_statistics, run_evt_gpd, run_sku_filter, run_param_summary
from src.hwk_bsts_forecasting.mle_bsts import run_backtest as run_hawkes_bsts
from src.causal_engine.causal import run_causal_engine
from src.backtest.backtest import run_quantitative_backtest

def call(fn, df, **kwargs):
    import inspect
    sig = inspect.signature(fn)
    state = kwargs.get("state", {}) or {}

    # use the enriched df stored by the previous step if available
    if "df" in state:
        df = state["df"]

    # drop state kwarg if the function does not declare it
    if "state" not in sig.parameters:
        kwargs.pop("state", None)

    result = fn(df, **kwargs)
    return result if isinstance(result, dict) else state

if __name__ == "__main__":
    print("\n ANOMALY DETECTION PIPELINE - SCENARIO 1 (GROSS)")
    scenario = 1
    cfg = SCENARIO_CONFIGS[scenario]
    df1 = load_and_clean_uci(scenario="gross")

    state = {}
    state = call(run_global_statistics, df1, scenario=scenario, state=state)
    state = call(run_evt_gpd, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_sku_filter, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_param_summary, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_mock_delays, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_var, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_monte_carlo, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_causal_engine, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_hawkes_bsts, df1, scenario=scenario, cfg=cfg, state=state)
    state = call(run_quantitative_backtest, df1, scenario=scenario, cfg=cfg, state=state)

    print("\n ANOMALY DETECTION PIPELINE - SCENARIO 2 (NETTED)")
    scenario = 2
    cfg = SCENARIO_CONFIGS[scenario]
    df2 = load_and_clean_uci(scenario="netted")

    state = {}
    state = call(run_global_statistics, df2, scenario=scenario, state=state)
    state = call(run_evt_gpd, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_sku_filter, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_param_summary, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_mock_delays, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_var, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_monte_carlo, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_causal_engine, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_hawkes_bsts, df2, scenario=scenario, cfg=cfg, state=state)
    state = call(run_quantitative_backtest, df2, scenario=scenario, cfg=cfg, state=state)
    
# COMPLETED
