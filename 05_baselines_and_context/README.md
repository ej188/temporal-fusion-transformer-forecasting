# Baselines And Context

This folder contains the compact public baseline/context layer for the TFT project.

The raw regression notebooks and generated artifacts were removed because they contained exploratory outputs and row-level identifiers. The public folder now keeps:

- `sarimax_baseline.py`: consolidated SARIMAX baseline script with volume-based and fill-rate-based modes.
- `baseline_summary.md`: concise explanation of the baseline approaches and why TFT became the final direction.
- `clustering_context.md`: how clustering and segmentation informed the forecasting problem.

The purpose of this folder is comparison and context, not to preserve every exploratory experiment.
