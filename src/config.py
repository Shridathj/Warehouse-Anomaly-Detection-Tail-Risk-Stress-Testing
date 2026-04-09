# src/config.py
import numpy as np

SCENARIO_CONFIGS = {
    1: { "label" : "Scenario 1 - Max Risk (NO Refunds)",
        "short_label" : "s1_gross",
        "SURGE_PCT": 0.18,
        "DRAGON_PCT": 0.00032,
        "DRAGON_BIAS_EXP": 1.35,
        "NORMAL_MEAN_MIN": 25.0,
        "SURGE_MEAN_MIN": 90.0,
        "DRAGON_MEAN_MIN": 420.0,
        "SIGMA_NORMAL": 0.50,
        "SIGMA_SURGE": 0.65,
        "SIGMA_DRAGON": 0.85,
        "NORMAL_CLIP": (5, 120),
        "SURGE_CLIP": (30, 480),
        "DRAGON_CLIP": (180, 1440),
        "SLA_BINS": [0, 30, 120, 480, np.inf],
        "SLA_LABELS": ["On-Time", "Minor Delay", "SLA Breach", "Fulfillment Failure"],
        "SLA_BREACH_MIN": 240,  # 4 hours
        "ANNUAL_HOLDING_RATE": 0.25,
        "GROSS_MARGIN": 0.30,
        "TARGET_ANNUAL_DRAGONS": 104,
        "VALUE_COL": "OrderValue_GBP",
        },
        
    2: { "label" : "Scenario 2 - Netted (Refunds/ Cancellations + Partial Cancellations)",
        "short_label" : "s2_netted",
        "SURGE_PCT": 0.18,
        "DRAGON_PCT": 0.00025,
        "DRAGON_BIAS_EXP": 1.20,
        "NORMAL_MEAN_MIN": 25.0,
        "SURGE_MEAN_MIN": 90.0,
        "DRAGON_MEAN_MIN": 360.0,
        "SIGMA_NORMAL": 0.50,
        "SIGMA_SURGE": 0.65,
        "SIGMA_DRAGON": 0.82,
        "NORMAL_CLIP": (5, 120),
        "SURGE_CLIP": (30, 360),
        "DRAGON_CLIP": (200, 1440),
        "SLA_BINS": [0, 30, 120, 360, np.inf],
        "SLA_LABELS": ["On-Time", "Minor Delay", "SLA Breach", "Fulfillment Failure"],
        "SLA_BREACH_MIN": 360,  # 6 hours   
        "ANNUAL_HOLDING_RATE": 0.25,
        "GROSS_MARGIN": 0.30,
        "TARGET_ANNUAL_DRAGONS": 72,
        "VALUE_COL": "OrderValue_GBP", 
        },
    }

# Backtest configuration 
BACKTEST_CONFIG = {
    "MIN_TRAIN_DAYS": 180,  # Minimum training window
    "WINDOW_STEP": 30,  # Purged expanding window step 
    "PURGE_GAP": 14,  # Gap between train cutoff and window end 
}

# Causal inference configuration 
CAUSAL_CONFIG = {
    "PSM_CALIPER": 0.005,  # Propensity score matching caliper
    "PSM_SUBSAMPLE": 50_000,  # PS fitting subsample size
    "QR_SUBSAMPLE": 100_000,  # Quantile regression subsample size
    "BOOTSTRAP_REPS": 500,  # Bootstrap replications
    "QUANTILES": [0.50, 0.75, 0.95, 0.99, 0.999],  # Quantiles for analysis
    "MAX_COUNTRIES": 8,  # Maximum countries to include in causal model
}

# Add backtest and causal config to each scenario
for scenario_id in [1, 2]:
    SCENARIO_CONFIGS[scenario_id].update(BACKTEST_CONFIG)
    SCENARIO_CONFIGS[scenario_id].update(CAUSAL_CONFIG)
        
    
    
           