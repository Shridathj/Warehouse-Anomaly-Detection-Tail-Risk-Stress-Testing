## Limitations & Trade-offs Important)

This is a public portfolio project built with only 374 days of open-source retail data.

- Backtest limited to ~6 windows due to short dataset and conservative purge (250-day train + 14-day gap). Coverage tests therefore have low statistical power.
- Monte-Carlo daily dragon revenue is i.i.d. lognormal for computational simplicity; the separate Hawkes section models clustering. Numbers are consistent but not yet integrated.
- PSM uses standard 1:1 caliper matching (no balanced diagnostics or double-robustness in this version).
- Quantile regression uses single random subsample (no bootstrap or clustering).
- Delays are synthetic (calibrated to WERC/CSCMP 2025 public summaries) because real WMS telemetry is proprietary.
- GPD and Hawkes fitted on full sample rather than rolling/purged.
- BSTS uses conservative fixed Q/R priors.

These are deliberate trade-offs given data constraints and time. For more robustness and accuracy, the following integrations would be made:
- Integrate Hawkes intensity directly into MC paths via thinning.
- Run daily rolling backtests + bootstrap.
- Add balance diagnostics, double-robust estimation, and clustering.
- Plug in real warehouse timestamps.

The core mathematics (EVT/GPD, causal identification, self-exciting processes, Kalman BSTS) remain correct and directly solve the dragon tail-risk problem.
