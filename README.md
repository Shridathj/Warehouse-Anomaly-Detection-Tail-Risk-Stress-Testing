# Warehouse Anomaly Detection & Tail-Risk Stress Tester

**Professional Project Summary & Technical Report**  
Developed by Pranav  
April 2026

## Abstract

This project quantifies the preventable financial loss arising from extreme high-value “dragon” orders (99th-percentile transactions) when warehouse fulfilment service levels degrade from the 99th to the 95th percentile. Using the UCI Online Retail dataset (541 000 raw rows, 3 665 SKUs, £8.9 M gross revenue over 374 days), two complementary scenarios were modelled: **gross (maximum exposure)** and **netted (realistic after cancellations)**. Realistic delays were simulated using industry parameters(educated guesses) from the 2025 WERC DC Measures Report and CSCMP State of Logistics Report. The end-to-end pipeline combines extreme-value theory, Monte-Carlo simulation, causal inference, Hawkes processes, and Bayesian Structural Time Series forecasting. Backtesting confirms model stability. Maintaining 99th-percentile SLA reduces exposure to near zero. All parameters and outputs are rigorously benchmarked and referenced.

## Summary

The Warehouse Anomaly Detection & Tail-Risk Stress Tester provides a framework for supply-chain leaders to measure and mitigate tail-risk exposure from high-value “dragon” orders. Key headline: in a bad year, relaxing service levels from the 99th to the 95th percentile creates **£77 000–£268 000** of preventable annual loss (netted to gross scenarios), while 99th-percentile fulfilment keeps exposure near zero. All results are validated by backtesting (shows acceptable violation rates under the stated assumptions). (Values are for directional purposes only).

## Note on backtesting results:

Scenario 1 (Gross): 0/34 violations (0.0%, Kupiec p=1.00, Christoffersen p=1.00).
Scenario 2 (Netted): 2/33 violations (6.1%, Kupiec p=0.79, Christoffersen p=0.61).
Reason:
Scenario 1 uses a harsher framing - no netting, stricter 4-hour SLA breach threshold (240 min vs 360 min), longer dragon delays (420 min vs 360 min), and stronger value bias (1.35 vs 1.20). This produces a heavier loss tail, so the model sets higher VaR/ES thresholds and appears “perfectly calibrated” (0 violations).
Scenario 2 uses realistic netting and more lenient parameters, resulting in a smoother loss distribution and a decent violation rate (6.1%). This backtest has decent statistical power and is the primary operational model (due to the small dataset, the statistical power of these tests are weak).

## Project Overview & Data

Owing to the proprietary nature of live warehouse transaction and fulfilment data, the publicly available UCI Online Retail dataset (2010–2011) was selected as the closest realistic proxy, with delays synthetically calibrated to 2025 industry benchmarks.
After cleaning (positive quantity & unit price, non-missing CustomerID, removal of miscellaneous codes), two parallel scenarios were created:

- **Scenario 1 (Maximum Exposure)**: Gross demand with no netting of refunds/cancellations.  
- **Scenario 2 (Realistic Netted Exposure)**: Full CustomerID–SKU netting to reflect actual fulfilled demand.

Value-biased synthetic delays (18 % surge + 0.025–0.032 % dragon tier) were overlaid using log-normal distributions calibrated to 2025 WERC/CSCMP benchmarks. Holding costs calculated at 25 % APR.

## Kaggle Notebook: https://www.kaggle.com/code/prnavjoshi/warehouse-anomaly-detection-stress-testing

## Detailed Methodology

The pipeline consists of nine rigorously linked stages:

1. **Data Ingestion & Pre-processing** – Filtering and CustomerID–StockCode netting.  
2. **Global Diagnostics & Pareto Filtering** (80/20 rule).  
3. **Extreme-Value Tail Modelling (EVT/GPD)** – Hill, moment, GEV estimators and peaks-over-threshold fitting.  
4. **Realistic Delay & Anomaly Simulation** – Tiered (normal/surge/dragon) log-normal delays.  
5. **Monte-Carlo Value-at-Risk & Expected-Shortfall** – 10 000 annual paths, VaR95/99, ES95.  
6. **Causal Validation** – Propensity-score matching + quantile regression.  
7. **Temporal Dependence & Forecasting** – Hawkes self-exciting process + Bayesian Structural Time Series.  
8. **Purged Expanding-Window Backtesting** – Kupiec and Christoffersen tests.  
9. **Reporting & Stress Interpretation** – Actionable mitigation dashboard.

## Backtesting Results & Financial Impact  

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
- `results/` – The plots and the expected numeric results of both scenarios are included
- To execute the full pipeline, install requirements 'pip install -r requirements.txt' and then, run `notebooks/run_src.py`

## Technologies

- **Core**: Python 3, pandas, NumPy, SciPy, statsmodels  
- **Advanced**: PyMC / PyStan (for BSTS), Hawkes process implementation, scikit-learn (PSM), extreme-value modelling libraries  
- **Visualisation**: Plotly, Matplotlib, Seaborn  

## Parameter Calibration & References

All synthetic parameters grounded in the latest 2025 WERC DC Measures Report and CSCMP State of Logistics Report. Full references included in the project files.

## Initial flawed attempts: Black-Scholes/Merton jump-diffusion for loss estimation (early overestimation ~$42M → discarded after math redo).

## Conclusion

This project demonstrates technical statistical and econometric techniques applied to a real-world supply-chain problem. It delivers immediate (simulated) operational value in retail, e-commerce, or 3PL environments.

This is an *undergraduate research/ portfolio project* built under real constraints.
All financial figures are under stated assumptions and are directional/illustrative only.  
*Do not treat the results as validated or suitable for operational decisions.*  
Full honest framing, limitations, and scope are documented in the updated summary:  
*project_report/updated_anomaly_summary.pdf*

---

**License**: Apache2.0
**Author**: Pranav  
**Last Updated**: April 2026
