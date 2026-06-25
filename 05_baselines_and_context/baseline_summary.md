# Baseline Summary

Before the final TFT workflow, the project explored several baseline and alternative forecasting directions.

## Regression Baselines

Early regression work used lagged order and shipment features to estimate future shipment quantities. These models were useful because they were simple, fast, and interpretable.

Limitations:

- Required manual feature construction for temporal behavior.
- Struggled with sparse and intermittent demand.
- Did not naturally share information across many related time series.
- Had limited ability to model long-range temporal dependencies.

## SARIMAX Baseline

SARIMAX was tested as a classical seasonal time-series baseline. The cleaned public script is `sarimax_baseline.py`.

The consolidated script includes:

- Monthly date parsing and duplicate aggregation.
- Supplier-part or facility-supplier-part series construction.
- Leakage-safe lag and rolling features.
- Seasonal SARIMAX with `(1,1,1)(1,1,1,12)`.
- Seasonal-naive fallback when SARIMAX fails to converge.
- MAE, MAPE, and WAPE evaluation.
- Two selection modes:
  - `volume`: model top-volume series.
  - `fill_rate`: rank suppliers by fulfillment quality, then model selected supplier-part series.

SARIMAX provided a useful benchmark but was less scalable for a large collection of heterogeneous sparse series.

## NeuralForecast And Other Experiments

Additional experiments included NHITS, neural forecasting variants, CatBoost, and fine-tuning/foundation-model exploration. These were valuable for comparison, but they were not kept as raw public files because they were exploratory and contained private paths or generated outputs.

## Why TFT Became The Main Model

TFT was the strongest fit for the final project goals because it can:

- Learn across many related series.
- Use static categorical identifiers.
- Combine known future inputs with historical target dynamics.
- Incorporate supplier, part, and global demand signals.
- Produce horizon-aware forecasts and interpretation artifacts.

The final public repo therefore presents SARIMAX and regression as context, while centering the TFT implementation and validation.
