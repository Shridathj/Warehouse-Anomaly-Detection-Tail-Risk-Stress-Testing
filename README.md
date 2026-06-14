# Warehouse Anomaly Detection & Tail-Risk Stress Tester

**Portfolio Project** | Supply Chain Risk & Quantitative Modelling

Developed by Pranav | April 2026 (Updated May 2026)

---

## Abstract

This project quantifies the **preventable financial loss** arising from extreme high-value “dragon” orders when warehouse fulfilment service levels degrade from the 99th to the 95th percentile.

Using the UCI Online Retail dataset and industry-calibrated parameters, it combines **Extreme Value Theory (EVT)**, **Monte-Carlo simulation**, **Hawkes processes**, custom **Bayesian Structural Time Series (BSTS)**, causal inference, and rigorous purged backtesting to measure tail-risk exposure in warehouse operations.

**Key Finding**: Maintaining 99th-percentile service levels on high-value orders can eliminate nearly all preventable tail-risk loss.

---

## Quick Start locally (or click on the Streamlit link below)

```bash
# 1. Clone and install
git clone https://github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing.git
cd Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing
pip install -r requirements.txt

# 2. Run the full pipeline locally (both scenarios) (**OPTIONAL**; 'results/ dashboard_cache/' directly loads the computed outputs and charts on to the streamlit dashborad)
python run_src.py

# 3. Interactive dashboard (for cloud deployment — see DEPLOYMENT.md)
streamlit run streamlit_dashboard.py **OR** python -m streamlit run streamlit_dashboard.py
```

Pipeline plots from `run_src.py` are written to `plots/scenario1/` and `plots/scenario2/` (folder is wiped at each `run_src.py` run).

> The pipeline executes global statistics, EVT/GPD modelling, delay simulation, VaR/ES, Monte-Carlo, causal analysis, Hawkes + BSTS forecasting, and quantitative backtesting.

## Streamlit link: https://warehouse-anomaly-detection-tail-risk-stress-testing-jaag2aynu.streamlit.app
---

## Project Structure

```
.
├── src/                      # Core modular Python package (do not modify lightly)
│   ├── config.py
│   ├── data/
│   ├── delay_simulation/
│   ├── global_statistics/
│   ├── risk/                 # VaR, ES, Monte-Carlo
│   ├── causal_engine/
│   ├── hwk_bsts_forecasting/ # Custom Hawkes MLE + state-space BSTS
│   ├── backtest/
│   └── utils/               
├── run_src.py                # Main entry point (orchestrates both scenarios)
├── project_report/           # Technical reports & PDFs
├── results/                  # Generated plots and outputs
├── dataset/                  # Raw & cleaned data
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Methodology

The end-to-end pipeline follows nine rigorously linked stages:

1. Data ingestion & cleaning
2. Global diagnostics & Pareto filtering
3. Extreme Value Theory (EVT/GPD) tail modelling
4. Realistic delay & anomaly simulation (industry-calibrated)
5. Monte-Carlo VaR & Expected Shortfall
6. Causal validation (PSM + quantile regression)
7. Hawkes process + custom BSTS forecasting
8. Purged expanding-window backtesting (Kupiec & Christoffersen tests)
9. Reporting & stress interpretation

All synthetic parameters are grounded in the **2025 WERC DC Measures Report** and **CSCMP State of Logistics Report**.

**Model Design Philosophy**  
The Monte Carlo engine is implemented as a **marginal benchmark** in the Loss Distribution Approach (LDA) tradition. It uses the empirical daily dragon revenue distribution (lognormal with observed CV, capped at 3.0) and a constant empirical breach rate. The Hawkes + BSTS engine serves as the **dynamic refinement**, explicitly modelling self-excitation in dragon arrivals and producing a forward-looking frequency forecast. The long-run intensity of the stationary Hawkes process,  
$$
\lambda_\infty = \frac{\mu}{1 - \alpha/\beta},
$$  
provides a natural anchor for unconditional expected loss. In the current dataset, the two engines produce closely aligned central estimates of annual preventable loss. This consistency validates calibration, while the modest gap (Hawkes slightly lower) is consistent with the dynamic model respecting stationary intensity.

---

## Key Results (Directional)

- **Preventable annual loss** at 95th-percentile SLA: **£77k – £268k**
- Maintaining **99th-percentile SLA** on high-value orders reduces exposure to near zero.
- Backtesting shows acceptable violation rates under stated assumptions.

> Full details, limitations, and methodology are available in `project_report/updated_anomaly_summary.pdf`.

---

## Reproducibility

- Primary execution: `python run_src.py`
- All random seeds and parameters are controlled via `src/config.py`.

---

## Technologies

- **Core**: Python ≥ 3.12, pandas, NumPy, SciPy, statsmodels
- **Advanced**: Custom Hawkes MLE + state-space BSTS (NumPy/Numba), scikit-learn, extreme-value modelling
- **Visualisation**: Plotly, Matplotlib, Seaborn

---

## Limitations & Scope

This is an undergraduate/portfolio project built under real data constraints. All financial figures are **directional and illustrative only**. See `LIMITATIONS.md` and the project report for full honest framing.

---

## License

Apache 2.0

---

**Author**: Pranav  
**Repository**: [github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing](https://github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing)
**Kaggle Notebook**: [kaggle.com/prnavjoshi/warehouse-anomaly-detection-stress-testing](https://www.kaggle.com/code/prnavjoshi/warehouse-anomaly-detection-stress-testing)
