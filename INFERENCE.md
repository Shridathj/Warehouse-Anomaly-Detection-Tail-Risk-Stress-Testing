# Inference and Enhancement Notes

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