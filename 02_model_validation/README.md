# Model Validation

This folder explains how the Temporal Fusion Transformer (TFT) was validated and where the validation logic lives in the implementation.

The key validation material is in `01_tft_model_implementation/train_tft.py`:

- The final 6 months are held out through `training_cutoff = data["time_idx"].max() - max_prediction_length`.
- The TFT horizon is `max_prediction_length = 6`.
- The encoder context is `max_encoder_length = 36`.
- Validation uses `TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)` so the validation loader mirrors the training schema but predicts the held-out horizon.
- Rolling and lag features use `shift(1)` before aggregation to avoid target leakage.
- Metrics are computed after reversing the `log1p` target transform with `torch.expm1`.

Files in this folder:

- `validate_tft_forecasts.py`: standalone CSV-based validator for forecast outputs.
- `validation_strategy.md`: how the train/validation split and leakage controls work.
- `metric_definitions.md`: how MAE, MAPE, per-horizon MAE, and WAPE are interpreted.
- `data_schema.md`: expected anonymized input schema and validation assumptions.

The public repository does not include the private dataset or trained checkpoints. These notes document the validation design so reviewers can understand the modeling rigor without needing access to confidential data.

Example standalone validation run:

```bash
python 02_model_validation/validate_tft_forecasts.py \
  --actuals path/to/anonymized_actuals.csv \
  --forecasts path/to/tft_forecasts.csv \
  --output validation_summary.csv
```
