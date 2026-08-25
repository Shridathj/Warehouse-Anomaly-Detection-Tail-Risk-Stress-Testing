"""
Tests for pipeline structure, configuration, and orchestration integrity.
"""

import inspect
import logging


def test_plotly_export_logging_suppressed():
    """Kaleido image export should not emit noisy INFO-level browser logs."""
    from src.utils.plot_paths import _configure_plotly_export_logging

    logger = logging.getLogger("kaleido")
    logger.setLevel(logging.INFO)

    _configure_plotly_export_logging()

    assert logger.level == logging.WARNING


def test_both_scenarios_exist(scenario_configs):
    """Verify that exactly two scenarios are defined."""
    assert len(scenario_configs) == 2
    assert 1 in scenario_configs
    assert 2 in scenario_configs

def test_scenario_labels(scenario_configs):
    """Check that scenarios have correct short labels."""
    assert scenario_configs[1]["short_label"] == "s1_gross"
    assert scenario_configs[2]["short_label"] == "s2_netted"

def test_required_config_keys(scenario_configs):
    """Ensure critical configuration keys exist in both scenarios."""
    required_keys = [
        "label",
        "short_label",
        "SURGE_PCT",
        "DRAGON_PCT",
        "SLA_BINS",
        "SLA_LABELS",
        "TARGET_ANNUAL_DRAGONS",
    ]

    for scenario_id in [1, 2]:
        cfg = scenario_configs[scenario_id]
        for key in required_keys:
            assert key in cfg, f"Missing key '{key}' in scenario {scenario_id}"


def test_pipeline_execution_sequence():
    """Verify the expected order of pipeline stages."""
    expected_order = [
        "run_global_statistics",
        "run_evt_gpd",
        "run_sku_filter",
        "run_param_summary",
        "run_mock_delays",
        "run_var",
        "run_monte_carlo",
        "run_causal_engine",
        "run_hawkes_kalman",
        "run_quantitative_backtest",
    ]

    # Import all functions
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

    functions = {
        "run_global_statistics": run_global_statistics,
        "run_evt_gpd": run_evt_gpd,
        "run_sku_filter": run_sku_filter,
        "run_param_summary": run_param_summary,
        "run_mock_delays": run_mock_delays,
        "run_var": run_var,
        "run_monte_carlo": run_monte_carlo,
        "run_causal_engine": run_causal_engine,
        "run_hawkes_kalman": run_hawkes_kalman,
        "run_quantitative_backtest": run_quantitative_backtest,
    }

    for step in expected_order:
        assert step in functions, f"Missing pipeline step: {step}"
        assert callable(functions[step]), f"{step} is not callable"


def test_pipeline_functions_have_df_parameter():
    """Check that core processing functions accept a DataFrame."""
    from src.global_statistics.global_stats import run_global_statistics
    from src.risk.var_es import run_var
    from src.risk.monte_carlo import run_monte_carlo

    for fn in [run_global_statistics, run_var, run_monte_carlo]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert len(params) > 0, f"{fn.__name__} has no parameters"

def test_run_src_py_exists(project_root):
    """Verify that the main entry point exists."""
    assert (project_root / "run_src.py").exists()

def test_requirements_file_exists(project_root):
    """Verify requirements.txt exists and is not empty."""
    req_file = project_root / "requirements.txt"
    assert req_file.exists()
    assert req_file.stat().st_size > 0