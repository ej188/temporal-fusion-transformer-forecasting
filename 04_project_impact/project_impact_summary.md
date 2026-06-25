# Project Impact Summary

## Problem

Supply-chain shipment planning is difficult when demand is spread across many supplier-part combinations, many of which are sparse, intermittent, or volatile. Traditional per-series forecasting approaches struggle because each individual series may not contain enough clean history to support a reliable standalone model.

The project addressed this by building a multi-series forecasting workflow that can learn shared structure across supplier-part histories while still preserving local patterns such as lagged shipments, rolling demand, supplier behavior, part behavior, and seasonality.

## Practical Value

The final TFT workflow supports a 6-month planning horizon. That horizon is useful because it gives planning teams enough time to identify potential shipment shortfalls, inspect supplier-part risk areas, and adjust inventory or procurement decisions before shortages become urgent.

The model was designed for operational realities:

- Intermittent demand, including zero-shipment months.
- High-cardinality supplier and part identifiers.
- Many related but heterogeneous time series.
- Changing seasonal and rolling demand patterns.
- Need for aggregate performance metrics that reflect business volume.

## Impact Of The TFT Approach

The Temporal Fusion Transformer improved the project in four ways:

1. It modeled thousands of related time series jointly rather than treating each supplier-part pair as a fully separate problem.
2. It incorporated static identifiers, time-varying order signals, lagged shipment history, rolling statistics, and hierarchical supplier/part/global demand features.
3. It supported interpretable diagnostics through variable importance and prediction plots.
4. It produced a strong aggregate validation result, with the best internal run reaching approximately 13.4% WAPE.

## Operational Interpretation

WAPE was used as the primary summary metric because it weights forecast errors by shipment volume. That matters in a supply-chain setting: a small error on a high-volume part can matter more than a large percentage error on a tiny or inactive series.

The model's outputs can support:

- Monthly shipment planning.
- Early identification of high-risk supplier-part combinations.
- Prioritization of planner attention toward high-volume or high-error areas.
- Scenario review when order behavior changes.
- More consistent forecasting coverage across sparse and dense series.

## Supporting Analyses

The original exploratory impact work looked at:

- Supplier fill-rate trends.
- Supplier shortage concentration.
- Months of available history by part.
- Facility-level order and shortage summaries.
- Patterns indicating where forecasting coverage is strongest or weakest.

Those analyses are not published as raw notebooks because they contained internal paths and row-level outputs. The public repo keeps the resulting story: forecasting performance matters most when it can be translated into planning decisions, supplier review, and shortage-risk prioritization.

## Portfolio Takeaway

This project demonstrates an end-to-end applied forecasting workflow: data preparation, leakage-safe feature engineering, TFT implementation, baseline comparison, validation, performance evaluation, and operational framing. The core contribution is not only the model choice, but the translation of a complex private dataset into a forecasting system that can be evaluated and explained in business terms.
