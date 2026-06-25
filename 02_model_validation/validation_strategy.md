# Validation Strategy

The TFT was validated as a forward-looking forecasting problem, not as a random row-level prediction task. This matters because random splits would leak future behavior from the same supplier-part series into training.

## Forecast Setup

- Forecast horizon: 6 months.
- Encoder history: up to 36 months.
- Series grain: supplier-part monthly series.
- Target: `QTY_APPLIED`, transformed with `log1p` during training.
- Known inputs: calendar/time index and order-related features.
- Unknown historical inputs: shifted shipment lags, rolling shipment statistics, and hierarchical demand signals.

## Holdout Design

The implementation sets:

```python
max_prediction_length = 6
training_cutoff = data["time_idx"].max() - max_prediction_length
```

Rows up to the cutoff are used for training. The validation dataset asks the model to predict the held-out tail window. This simulates the operational use case: train on known history, then forecast the next planning window.

## Leakage Controls

The most important validation safeguard is that all target-derived features are shifted before use:

- `ship_lag1`, `ship_lag2`, `ship_lag3` use grouped `shift`.
- `ship_roll3` uses `shift(1).rolling(...)`.
- Supplier-level volume signals are aggregated by supplier-month, then shifted before rolling.
- Part-level volume signals are aggregated by part-month, then shifted before rolling.
- Global demand is shifted by one time step.

This means the feature row for month `t` only sees information available before month `t`.

## Baseline Check

The training script includes a naive baseline via PyTorch Forecasting's `Baseline`. This is useful as a sanity check: the TFT should outperform a simple carry-forward style baseline before its added complexity is justified.

## What Is Not Published

The private validation dataset, trained weights, per-series prediction records, and raw generated reports are not included. Public validation evidence is therefore described through methodology and aggregate metrics rather than redistributing private data.
