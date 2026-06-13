## Limitations & Trade-offs 

This is a public portfolio project built with only 374 days of open-source retail data.

- Monte Carlo vs Hawkes+BSTS: The Monte Carlo simulates daily dragon revenue under an i.i.d. lognormal assumption for computational simplicity and to generate a full loss distribution. The Hawkes+BSTS component separately models temporal clustering and produces a point forecast of annual loss. While the two approaches rest on different assumptions, they produce closely aligned central estimates of annual preventable loss. This convergence provides a degree of model triangulation; however, a fully integrated approach remains future work.
- PSM uses standard 1:1 caliper matching (no balanced diagnostics or double-robustness in this version).
- Quantile regression uses single random subsample (no bootstrap or clustering).
- Delays are synthetic (calibrated to WERC/CSCMP 2025 public summaries) because real WMS telemetry is proprietary.
- GPD and Hawkes fitted on full sample rather than rolling/purged.
- BSTS uses conservative fixed Q/R priors (basic implementaion).

These are deliberate trade-offs given data constraints, compute and time. For more robustness and accuracy, the following integrations would be made:
- A fully integrated approach e.g. Hawkes-driven simulation or propagating BSTS uncertainty.
- Run daily rolling backtests + bootstrap.
- Add balance diagnostics, double-robust estimation, and clustering.
- Plug in real warehouse timestamps.

The core mathematics (EVT/GPD, causal identification, self-exciting processes, Kalman BSTS) remain correct and directly solve the dragon tail-risk problem.
