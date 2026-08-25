# Warehouse Anomaly Detection & Tail-Risk Stress Testing

**Portfolio project** · Supply-chain risk · Quantitative modelling  
Pranav · April 2026 (updated August 2026)

Stress-tests **preventable fulfilment loss** on high-value “dragon” orders when warehouse service degrades from a **99th-percentile SLA** to a **95th-percentile / reduced QoS** regime.

The ticket file is the [UCI Online Retail](https://doi.org/10.24432/C5BW33) extract (1 Dec 2010 – 9 Dec 2011). It is treated as a **real retail order book**. It has no pick-complete time, no SLA clock, and no dragon label. Fulfilment delays are therefore **simulated** from log-normals whose means, dispersion and clips are calibrated to public summaries of the **2025 WERC DC Measures** and **CSCMP State of Logistics** reports — an **average-retail** delay regime, not a live WMS extract.

Sterling figures are **directional outputs under those assumptions**. They are not operational loss forecasts.

**Dashboard:** [warehouse-anomaly-detection-tail-risk-stress-testing.streamlit.app](https://warehouse-anomaly-detection-tail-risk-stress-testing.streamlit.app/)

---

## What is observed, and what is assigned

| Object | Status |
|---|---|
| Invoice, SKU, quantity, price, customer, country, timestamp | Observed |
| Fulfilment delay, SLA label, dragon / surge tier | **Assigned** (value-biased sample + log-normal delays) |
| Unfulfilled dragon | Dragon whose simulated delay clears the breach threshold (4 h Scenario 1, 6 h Scenario 2) |

Dragons are a thin, **value-biased** sample (~3.2 bps of tickets gross, ~2.5 bps after netting), not a detector output. The 99th→95th language is **SLA / QoS degradation** on that expensive tail: configured dragon delays sit in a degraded service band relative to a tight on-time SLA. It is not a causal estimate of a live distribution centre.

The **loss object is order value and unfulfilment**. Quantity EVT is a **domain diagnosis** (the demand tail is Fréchet; do not winsorise dragons).

---

## Two exposure views

| | Scenario 1 — gross | Scenario 2 — netted |
|---|---|---|
| Map | Positive-quantity tickets after dropping returns | Net at `(CustomerID, SKU)` over the year; keep the max-pick row |
| Question | Booked demand the warehouse must provision against | Demand that survives cancellations |
| Why both | The 80,995-unit PAPER CRAFT ticket is booked then reversed within minutes. Gross is short-horizon operational exposure; net is the P&L residual |

---

## Pipeline

Nine stages, in order. Later sterling numbers are **functionals of earlier objects**.

1. **Ingest & clean** — UCI file, two maps above (`src/data/loader.py`).
2. **Global structure** — moments, Pareto 80/20 on high-volume SKUs, daily ACF / Ljung–Box, Gaussianity tests. Serial dependence is **left in the series**: Hill / GEV / POT remain i.i.d. estimators, used here to show that the data are both **dependent and extreme**.
3. **EVT on quantity** — Hill, Dekkers–Einmahl–de Haan moment, GEV block maxima, mean-excess slope. **Diagnosis only** (Fréchet, \(\xi \approx 0.5\)). An AMSE curve is drawn and **not** used to fit a GPD on quantity.
4. **Delay simulation** — value-weighted dragons, uniform surge, independent log-normal delays, SLA bins (`src/config.py`).
5. **Holding cost & unfulfilment** — linear carrying cost at 25% APR; unfulfilled-dragon revenue is the preventable piece.
6. **Monte Carlo (LDA reference)** — i.i.d. log-normal daily dragon revenue × empirical breach rate × 30% margin. Marginal benchmark in the Loss Distribution Approach tradition. Hawkes clustering is **not** injected into these paths.
7. **Causal contrasts** — surge ATE on net revenue is a **negative control** (surge is assigned, not a cause of value). The object of interest is the **quantile contrast of the dragon grouping on order value**.
8. **Hawkes + Kalman** — exponential Hawkes MLE on dragon times; twelve-month count from a **Kalman local-linear-trend filter**. A Hawkes-modulated Monte Carlo is [future work](LIMITATIONS.md).
9. **Purged expanding-window backtest** — window-matched VaR, Kupiec and Christoffersen. GPD MLE is fitted here, on **daily realised loss**.

Full derivations, identities and implementation notes: [the monograph](project_report/A_Comprehensive_Approach_to_Tail_Risk_Estimation_via_EVT.pdf).

---

## How to read the sterling numbers

They answer different questions. Do not add them.

| Functional | What it is | Typical headline (directional) |
|---|---|---|
| LDA Monte Carlo ES95 (`src/risk/monte_carlo.py`) | One-year margin-adjusted preventable loss; **all calendar days** in the daily mean | ~£60k gross / ~£12k netted |
| Causal “annual impact” | Same identity: \(g \times (365/\Delta) \times\) unfulfilled-dragon revenue | ~£43k / ~£9k |
| Last-window backtest MC ES95 | Same path equation, but the daily mean is over **dragon-active days** in the last train slice | ~£248k / ~£77k |
| GPD “VaR95 / ES95” on daily loss | POT above the **99th** percentile of daily realised loss; printed 95% VaR is clamped to that threshold, so ES95 is a **far-tail daily mean excess**, not a 95% daily VaR | daily, not annual |
| \(p_{99}(\ell_t) \times 252\) | Stress exhibit (a bad day, every trading day). Not an expectation | — |

The older one-line range **£77k–£268k** is the last-window backtest ES95 pair. It is the **aggressive / bad-year** exhibit, not a revision of the LDA ~£60k / ~£12k.

“Zero anomalies → £0” and the reroute / safety-stock rows are **proportional-reduction arithmetic** on a count, not estimated treatment effects of holding 99th-percentile service.

---

## Backtest

The backtest module is a **replay of the delay DGP**, not a continuation of the delay-stage frame. That is why its coverage table is internally consistent with the printed engine.

Read it with the following in mind:

- **Window-matched VaR** compares a short test slice to a short-horizon MC quantile, not to an annual VaR. That is why exception counts stay small.
- Test loss is \(V+H\) (gross); window VaR is margin-adjusted dragon revenue. A 0/34 violation rate is **not** a calibration theorem.
- **Kupiec p = 1 at zero violations** overstates the evidence. With ~34 weekly windows and \(p=0.05\), zero exceptions is unsurprising (expected ≈ 1.7). The correct LR at \(k=0\) is a borderline non-rejection (~0.06), not a perfect score. Power is low on one year of tickets.
- **GPD threshold = 99th percentile of daily loss**, then the print asks for 95% VaR/ES. Unconstrained 95% sits below \(u\); the clamp makes “VaR95” = \(u\) and “ES95” the GPD mean above the 99th. Interpretable as **how bad a tail day is**, not as 95% daily VaR.
- Last-window ES95 uses an **active-day mean**. The gap versus the LDA ES95 is a mean definition, not a discovery that the last window was four times riskier.
- The P&L row that multiplies an already-margined MC ES by 30% again is a **second factor of \(g\)**. The headline to read is the pre-double-count MC ES95.

---

## Quick start

```bash
git clone https://github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing.git
cd Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing
pip install -r requirements.txt

# optional — full pipeline for both scenarios (wipes plots/scenario1 and plots/scenario2)
python run_src.py

# dashboard (uses results/dashboard_cache if present)
streamlit run streamlit_dashboard.py
```

Seeds and SLA / delay / margin parameters live in `src/config.py`. Cloud deploy: [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Layout

```
src/
  config.py
  data/                    # UCI load, gross vs netted maps
  delay_simulation/        # value-biased dragons, log-normal delays, SLA
  global_statistics/       # moments, Pareto filter, quantity EVT diagnosis
  risk/                    # holding / unfulfilment, LDA Monte Carlo
  causal_engine/           # surge ATE (negative control), dragon QTE
  hwk_kalman_forecasting/  # Hawkes MLE + Kalman local linear trend
  backtest/                # purged windows, daily-loss GPD, Kupiec / Christoffersen
  utils/
run_src.py
streamlit_dashboard.py
project_report/A_Comprehensive_Approach_to_Tail_Risk_Estimation_via_EVT.pdf
LIMITATIONS.md
INFERENCE.md
```

---

## Stack

Python ≥ 3.12 · pandas · NumPy · SciPy · statsmodels · Numba (Hawkes intensity) · scikit-learn · Plotly / Matplotlib / Seaborn · Streamlit

---


Details: [LIMITATIONS.md](LIMITATIONS.md), [INFERENCE.md](INFERENCE.md), and the [monograph](project_report/A_Comprehensive_Approach_to_Tail_Risk_Estimation_via_EVT.pdf).

---

Apache 2.0

**Author:** Pranav  
**Repo:** [github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing](https://github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing)  
**Kaggle:** [prnavjoshi/warehouse-anomaly-detection-stress-testing](https://www.kaggle.com/code/prnavjoshi/warehouse-anomaly-detection-stress-testing)
