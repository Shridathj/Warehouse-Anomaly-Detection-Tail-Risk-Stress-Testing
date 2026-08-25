"""
Test that all critical modules and functions can be imported successfully.
"""

class TestCoreImports:

    def test_config_import(self):
        from src.config import SCENARIO_CONFIGS
        assert isinstance(SCENARIO_CONFIGS, dict)
        assert 1 in SCENARIO_CONFIGS and 2 in SCENARIO_CONFIGS

    def test_data_loader_import(self):
        from src.data.loader import load_and_clean_uci
        assert callable(load_and_clean_uci)

    def test_delay_simulation_import(self):
        from src.delay_simulation.delays import run_mock_delays
        assert callable(run_mock_delays)

    def test_risk_modules_import(self):
        from src.risk.var_es import run_var
        from src.risk.monte_carlo import run_monte_carlo
        assert callable(run_var)
        assert callable(run_monte_carlo)

    def test_global_statistics_import(self):
        from src.global_statistics.global_stats import (
            run_global_statistics,
            run_evt_gpd,
            run_sku_filter,
            run_param_summary,
        )
        assert callable(run_global_statistics)
        assert callable(run_evt_gpd)
        assert callable(run_sku_filter)
        assert callable(run_param_summary)

    def test_forecasting_import(self):
        from src.hwk_kalman_forecasting.mle_kalman import run_backtest as run_hawkes_kalman
        assert callable(run_hawkes_kalman)

    def test_causal_engine_import(self):
        from src.causal_engine.causal import run_causal_engine
        assert callable(run_causal_engine)

    def test_backtest_import(self):
        from src.backtest.backtest import run_quantitative_backtest
        assert callable(run_quantitative_backtest)