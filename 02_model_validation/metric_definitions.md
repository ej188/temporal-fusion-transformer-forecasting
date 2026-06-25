# Metric Definitions

The TFT validation reports error in real shipment units after reversing the `log1p` target transform:

```python
y_true_real = torch.expm1(y_true)
y_pred_real = torch.expm1(y_pred)
```

## MAE

Mean Absolute Error measures the average absolute error in shipment units.

Why it matters:

- Easy to interpret operationally.
- Stable when actual demand is zero.
- Useful for per-horizon comparison.

## Per-Horizon MAE

The model predicts a 6-month horizon. Per-horizon MAE computes a separate MAE for month 1, month 2, through month 6.

Why it matters:

- Forecasts usually get harder farther into the future.
- A single average can hide horizon-specific degradation.
- Planning teams can see whether near-term or long-term forecasts are more reliable.

## MAPE

Mean Absolute Percentage Error is computed only where actual demand is greater than zero.

Why it needs a guard:

- Supply-chain data can be intermittent.
- Zero actual demand makes percentage error undefined.
- The implementation skips zero-actual rows for MAPE and reports `nan` if no valid rows exist.

## WAPE

Weighted Absolute Percentage Error is:

```text
sum(abs(actual - predicted)) / sum(actual) * 100
```

Why it is the primary project metric:

- Handles intermittent demand better than naive MAPE.
- Weighs high-volume series by their business impact.
- Gives an aggregate view of forecast quality across many supplier-part combinations.

The best internal TFT run achieved approximately 13.4% WAPE on the project validation data. Because the underlying data is private, this repository presents that number as an aggregate project result rather than publishing row-level forecast records.
