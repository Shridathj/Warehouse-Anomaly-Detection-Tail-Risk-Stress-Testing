# Warehouse Anomaly Detection & Tail-Risk Stress Tester

**Professional Project Summary & Technical Report**  
Developed by Pranav  
April 2026

## Abstract

This project quantifies the preventable financial loss arising from extreme high-value “dragon” orders (99th-percentile transactions) when warehouse fulfilment service levels degrade from the 99th to the 95th percentile. Using the UCI Online Retail dataset (541 000 raw rows, 3 665 SKUs, £8.9 M gross revenue over 374 days), two complementary scenarios were modelled: **gross (maximum exposure)** and **netted (realistic after cancellations)**. Realistic delays were simulated using industry-calibrated parameters from the 2025 WERC DC Measures Report and CSCMP State of Logistics Report. The end-to-end pipeline combines extreme-value theory, Monte-Carlo simulation, causal inference, Hawkes processes, and Bayesian Structural Time Series forecasting. Backtesting confirms model stability. Maintaining 99th-percentile SLA reduces exposure to near zero. All parameters and outputs are rigorously benchmarked and referenced.

## Summary

The Warehouse Anomaly Detection & Tail-Risk Stress Tester provides a complete, auditable framework for supply-chain leaders to measure and mitigate tail-risk exposure from high-value “dragon” orders. Key headline: in a bad year, relaxing service levels from the 99th to the 95th percentile creates **£77 000–£268 000** of preventable annual loss (netted to gross scenarios), while 99th-percentile fulfilment keeps exposure near zero. All results are validated by perfect backtesting calibration.

## Project Overview & Data

The analysis uses the real UCI Online Retail dataset. After cleaning (positive quantity & unit price, non-missing CustomerID, removal of miscellaneous codes), two parallel scenarios were created:

- **Scenario 1 (Maximum Exposure)**: Gross demand with no netting of refunds/cancellations.  
- **Scenario 2 (Realistic Netted Exposure)**: Full CustomerID–SKU netting to reflect actual fulfilled demand.

Value-biased synthetic delays (18 % surge + 0.025–0.032 % dragon tier) were overlaid using log-normal distributions calibrated to 2025 WERC/CSCMP benchmarks. Holding costs calculated at 25 % APR.

## Detailed Methodology

The pipeline consists of nine rigorously linked stages:

1. **Data Ingestion & Pre-processing** – Filtering and CustomerID–StockCode netting.  
2. **Global Diagnostics & Pareto Filtering** (80/20 rule).  
3. **Extreme-Value Tail Modelling (EVT/GPD)** – Hill, moment, GEV estimators and peaks-over-threshold fitting.  
4. **Realistic Delay & Anomaly Simulation** – Tiered (normal/surge/dragon) log-normal delays.  
5. **Monte-Carlo Value-at-Risk & Expected-Shortfall** – 10 000 annual paths, VaR95/99, ES95.  
6. **Causal Validation** – Propensity-score matching + quantile regression.  
7. **Temporal Dependence & Forecasting** – Hawkes self-exciting process + Bayesian Structural Time Series.  
8. **Purged Expanding-Window Backtesting** – Kupiec and Christoffersen tests (zero violations, p = 1.0000).  
9. **Reporting & Stress Interpretation** – Actionable mitigation dashboard.

## Backtesting Results & Financial Impact

Both scenarios produced **zero violations** with perfect statistical calibration.  

**Warehouse Management Implications**  
- Realistic (netted) operations: expected annual preventable dragon loss of **£77,459** (MC ES95) at 95th-percentile SLA.  
- Gross view: exposure rises to **£268,153** annually.  
- 5-year cumulative bleed without action: **£387 000–£1.34 M**.  
- Prioritising 99th-percentile SLA on high-value orders eliminates virtually all preventable tail loss.  
- Targeted interventions (fast lanes, dedicated safety stock) deliver immediate ROI.

## Repository Contents

- `src/` – Complete Python pipeline (data ingestion to forecasting)  
- `notebooks/` – Exploratory analysis and interactive visualisations  
- `data/` – Raw and cleaned UCI Online Retail dataset  
- `reports/` – Full technical report, backtesting results, and dashboards  
- `results/` – VaR/ES outputs and mitigation scenarios  

## Technologies

- **Core**: Python 3, pandas, NumPy, SciPy, statsmodels  
- **Advanced**: PyMC3 / PyStan (for BSTS), Hawkes process implementation, scikit-learn (PSM), extreme-value modelling libraries  
- **Visualisation**: Plotly, Matplotlib, Seaborn  

## Parameter Calibration & References

All synthetic parameters grounded in the latest 2025 WERC DC Measures Report and CSCMP State of Logistics Report. Full references included in the project files.

## Conclusion

This project demonstrates technical mastery of advanced statistical and econometric techniques applied to a real-world supply-chain problem. It delivers immediate operational value in retail, e-commerce, or 3PL environments.

---

**License**: Apache2.0
**Author**: Pranav  
**Last Updated**: April 2026
