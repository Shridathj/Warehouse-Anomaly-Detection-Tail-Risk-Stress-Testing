# Inference and Enhancement Notes

## Causal Engine Memory Behaviour (Scenario 1 vs Scenario 2)

The causal engine is the memory-heavy stage of the pipeline. On the gross book (scenario 1) it peaks near 14 GB. On the netted book (scenario 2) the same stage stays small. The difference is not the raw ticket count. It is how the two paths build intermediate frames.

UCI Online Retail is 541,909 invoice lines. Scenario 1 keeps every positive-quantity ticket after returns are dropped, so the frame remains hundreds of thousands of rows and the UK dominates. Scenario 2 nets at `(CustomerID, SKU)` and keeps the max-pick row, so the frame is much smaller. The causal stage then does three things that scale badly only in scenario 1.

**Propensity score.** Both scenarios fit the logistic model on a 50,000-row subsample (`PSM_SUBSAMPLE`). Scenario 1 then scores the full frame in one `predict_proba` call. Scenario 2 scores in chunks of 50,000 and deletes the chunks.

**Propensity-score matching.** Scenario 1 bins the propensity score with `pd.qcut` (50 bins) separately on the treated and control sets, then `pd.merge`s on the bin label inside each country. A merge on a shared bin is a cartesian product of that bin before the 0.005 caliper is applied. For a UK-sized country (roughly 65,000 treated at the 18 % surge rate, 295,000 control) that is on the order of 60 million candidate pairs before the filter. That intermediate frame is the memory spike. Scenario 2 never builds it: it sorts and uses `np.searchsorted` to inspect only the three nearest controls, marks used controls, and stops. It also caps countries at `MAX_COUNTRIES = 8`.

**Quantile regression.** Scenario 1 calls `QuantReg.from_formula` on up to 180,000 rows (`QR_SUBSAMPLE`) with `C(Country)` and `C(Hour)`. Patsy plus the Epanechnikov-kernel IRLS keeps dense design matrices and intermediate frames, and each of the two quantiles (0.95, 0.999) stores a full result. Scenario 2 builds the design matrix by hand, bins hour into four periods, uses `float32` dummies, and explicitly deletes and collects.

Backtesting is not the source. It is a purged expanding window (90-day minimum, 7-day step, 3-day purge) with window-matched VaR, Kupiec and Christoffersen tests, and a GPD on daily realised loss. That path is small.

### Proposed improvements

The causal engine is memory-traffic-bound, not compute-bound. The useful changes are the ones a compute-architecture or GPU-architecture exercise would actually measure.

- Replace the scenario-1 bin-merge with the scenario-2 nearest-neighbour scan, or with a short C++/CUDA kernel that emits only pairs inside the caliper. The match should never materialise the cartesian product.
- Cap or hash the country set in scenario 1 the way scenario 2 already does, and score propensity in chunks.
- Shrink the quantile design matrix (bin hour, drop rare countries) or replace `from_formula` with an explicit array so the intermediate frames are visible and deletable.
- Treat the Monte Carlo paths and the propensity scoring as the embarrassingly parallel kernels. Profile them for bandwidth and occupancy before adding more model complexity.

These are the same constraints that show up on a 6G radio unit: a decision has to finish inside a tight power and latency budget, and an intermediate frame that does not fit is a failed design, not a tuning problem.

## Proposed WMS KPI Dashboard Layer

Standard Warehouse Management System (WMS) Key Performance Indicators (KPIs) would further strengthen the project's applicability to real-world fulfillment operations. The following metrics are widely used by supply chain organisations to measure execution performance:

- Order cycle time
- Perfect order rate
- Fill rate
- Dock-to-stock time
- Inventory accuracy
- Pick accuracy
- SLA attainment

Incorporating these KPIs into the dashboard would enable direct linkage between tail-risk events (such as dragon orders and service-level degradation) and their impact on core operational performance. This would extend the current risk-focused analysis into a more comprehensive performance-and-risk view, which is relevant to WMS implementation, modernisation, and optimisation work.

These metrics were identified through review of publicly shared content by supply chain leaders and companies on professional platforms, as well as open-source supply chain and WMS resources. Full implementation of a production-grade WMS KPI layer is not feasible within this portfolio project, as it would require access to proprietary client WMS data, system configurations, and live operational telemetry that are not available in the public UCI Retail dataset.

## Data Quality Observation from UCI Retail Dataset (Scenario 1)

In Scenario 1, dragon orders were defined using all positive quantity entries after dropping negative (return/cancellation) records. Analysis of the raw data reveals multiple instances where very large positive quantity entries are followed within a short time window (approximately 12 minutes in observed cases) by an equal negative quantity entry of the same magnitude. One notable example involves an entry of approximately 80,000 units immediately followed by a -80,000 unit entry.

This recurring pattern supports two primary inferences:

1. There may be underlying issues in the order entry or system integration process that generate false large positive orders, which are subsequently corrected through cancellation or refund entries. This pattern raises data governance issues and would require validation and cleansing in a production WMS environment.

2. If the entries reflect actual business activity rather than system errors, the operation is accepting significant gross exposure to fulfil additional order volume. In this case, the business must provision inventory, capacity, and working capital for the full exposed amount before any netting occurs.

Both interpretations highlight the importance of robust data validation and exposure management when handling high-value or high-quantity orders in fulfillment systems.
