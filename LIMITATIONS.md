## Limitations & Trade-offs 

This is a public portfolio project built with only 374 days of open-source retail data.

- **Monte Carlo vs Hawkes+BSTS**: The Monte Carlo functions as a **marginal benchmark** in the Loss Distribution Approach (LDA) tradition. It takes the empirical distribution of daily dragon revenue (fitted as lognormal with observed CV, capped at 3.0) and applies a constant empirical breach rate. Because it is calibrated directly to observed daily aggregates, it implicitly incorporates historical clustering effects into its marginal moments rather than modelling arrival dynamics explicitly.

  The Hawkes + BSTS engine serves as the **dynamic refinement**. It explicitly captures self-excitation in dragon arrivals through the fitted Hawkes process and generates a forward-looking frequency forecast via BSTS. A key property of the stationary Hawkes process is that its long-run intensity converges to
  $$
  \lambda_\infty = \frac{\mu}{1 - \alpha/\beta},
  $$
  provided $\alpha/\beta < 1$ (which holds in both scenarios). This stationary intensity provides a natural anchor for unconditional expected frequency.

  In the current dataset, the two approaches produce closely aligned central estimates of annual preventable loss, with only modest differences (Hawkes yielding a slightly lower value). This consistency is reassuring: despite fundamentally different treatments of temporal dependence, both engines recover similar unconditional expected loss levels. The small gap indicates that, under the present data characteristics and loss definition, the marginal distribution of dragon revenue and average loss-given-breach dominate the annual aggregate more than the fine structure of arrival clustering.

  A fully integrated **Hawkes-modulated Monte Carlo** (in which daily intensity is driven by the fitted Hawkes process) remains future work and would be especially valuable on cleaner, proprietary WMS data where clustering effects are stronger.

- PSM uses standard 1:1 caliper matching (no balanced diagnostics or double-robustness in this version).
- Quantile regression uses single random subsample (no bootstrap or clustering).
- Delays are synthetic (calibrated to WERC/CSCMP 2025 public summaries) because real WMS telemetry is proprietary.
- GPD and Hawkes fitted on full sample rather than rolling/purged.
- BSTS uses conservative fixed Q/R priors (basic implementation).

These are deliberate trade-offs given data constraints, compute and time. For more robustness and accuracy, the following integrations would be made:
- A fully integrated approach e.g. Hawkes-driven simulation or propagating BSTS uncertainty.
- Run daily rolling backtests + bootstrap.
- Add balance diagnostics, double-robust estimation, and clustering.
- Plug in real warehouse timestamps.

The core mathematics (EVT/GPD, causal identification, self-exciting processes, Kalman BSTS) remain correct and directly solve the dragon tail-risk problem.
