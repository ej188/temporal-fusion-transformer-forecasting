#!/usr/bin/env python3
"""
Production-ready TFT pipeline (inference/training) with:
 - no leakage (all rolling features use shift(1))
 - order lag features
 - supplier/part/global hierarchical signals
 - robust LR finder fallback
 - metrics: MAE, per-horizon MAE, MAPE (zero-guarded), WAPE
 - conservative model size (safe defaults) with comments where to change
"""

import os
import glob
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, Baseline
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from pytorch_forecasting.metrics import MAE

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
# ---------------------------
# Config
# ---------------------------
parser = argparse.ArgumentParser(description="Train and validate a Temporal Fusion Transformer shipment forecast model.")
parser.add_argument("--data", required=True, help="Path to an anonymized monthly shipment/order CSV.")
parser.add_argument("--report-pdf", default="tft_forecast_report.pdf", help="Output path for the validation report PDF.")
parser.add_argument("--model-dir", default="final_models", help="Directory for checkpoints and training artifacts.")
args = parser.parse_args()

DATA_PATH = args.data
REPORT_PDF = args.report_pdf
MODEL_DIR = args.model_dir
os.makedirs(MODEL_DIR, exist_ok=True)

LOG = logging.getLogger("tft_pipeline")
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# safe device settings
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
LOG.info("TFT forecasting pipeline")
# ---------------------------
# Load & basic cleaning
# ---------------------------
LOG.info("STEP 1: LOAD & CLEAN")
data = pd.read_csv(DATA_PATH)
data["MONTHYEAR"] = pd.to_datetime(data["MONTHYEAR"], errors="coerce")
data = data.dropna(subset=["MONTHYEAR"]).copy()
# time machine guard
data = data[data["MONTHYEAR"] < "2025-01-01"].copy()

# basic numeric cleaning
data["ORDERED_QTY"] = data["ORDERED_QTY"].fillna(0).clip(lower=0)
data["QTY_APPLIED"] = data["QTY_APPLIED"].fillna(0).clip(lower=0)

# time index
min_date = data["MONTHYEAR"].min()
data["time_idx"] = ((data["MONTHYEAR"].dt.year - min_date.year) * 12 +
                    (data["MONTHYEAR"].dt.month - min_date.month)).astype(int)

# target and logged ordered qty
data["log_ship_qty"] = np.log1p(data["QTY_APPLIED"]).astype(np.float32)
data["log_ordered_qty"] = np.log1p(data["ORDERED_QTY"]).astype(np.float32)

# ensure grouping order before shift-based features
group_ids = ["SUPP_CD_HASHED", "CAT_ID_NO_HASHED"]
data = data.sort_values(group_ids + ["time_idx"]).reset_index(drop=True)

# ---------------------------
# Feature engineering (no leakage)
# ---------------------------
LOG.info("STEP 1.5: FEATURE ENGINEERING (PAST-ONLY / ROLLING)")

# 1. Group-Level Lags and Rolling (Safe because grouped by BOTH Supp and Part)
data["ship_lag1"] = data.groupby(group_ids)["log_ship_qty"].shift(1).fillna(0).astype(np.float32)
data["ship_lag2"] = data.groupby(group_ids)["log_ship_qty"].shift(2).fillna(0).astype(np.float32)
data["ship_lag3"] = data.groupby(group_ids)["log_ship_qty"].shift(3).fillna(0).astype(np.float32)
data["ship_roll3"] = data.groupby(group_ids)["log_ship_qty"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0).astype(np.float32)

data["order_lag1"] = data.groupby(group_ids)["ORDERED_QTY"].shift(1).fillna(0).astype(np.float32)
data["order_roll3"] = data.groupby(group_ids)["ORDERED_QTY"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0).astype(np.float32)
data["log_order_lag1"] = np.log1p(data["order_lag1"]).astype(np.float32)
data["log_order_roll3"] = np.log1p(data["order_roll3"]).astype(np.float32)

# 2. Supplier-Level Features (Must aggregate first to prevent cross-part contamination)
# Sum up all parts for a supplier in a given month
supp_monthly = data.groupby(["SUPP_CD_HASHED", "time_idx"])["QTY_APPLIED"].sum().reset_index()
supp_monthly = supp_monthly.sort_values(["SUPP_CD_HASHED", "time_idx"])
supp_monthly["supp_log_qty"] = np.log1p(supp_monthly["QTY_APPLIED"])
# Now shift and roll safely (since there is only 1 row per supplier per month)
supp_monthly["avg_vol_supplier"] = supp_monthly.groupby("SUPP_CD_HASHED")["supp_log_qty"].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
supp_monthly["supplier_roll_mean6"] = supp_monthly["avg_vol_supplier"] # Keeping your naming convention
supp_monthly["supplier_trend"] = supp_monthly.groupby("SUPP_CD_HASHED")["supplier_roll_mean6"].diff().fillna(0)

# 3. Part-Level Features (Aggregate across facilities if facilities existed, but safe to do anyway)
part_monthly = data.groupby(["CAT_ID_NO_HASHED", "time_idx"])["QTY_APPLIED"].sum().reset_index()
part_monthly = part_monthly.sort_values(["CAT_ID_NO_HASHED", "time_idx"])
part_monthly["part_log_qty"] = np.log1p(part_monthly["QTY_APPLIED"])
part_monthly["avg_vol_part"] = part_monthly.groupby("CAT_ID_NO_HASHED")["part_log_qty"].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
part_monthly["part_roll_mean6"] = part_monthly["avg_vol_part"]
part_monthly["part_trend"] = part_monthly.groupby("CAT_ID_NO_HASHED")["part_roll_mean6"].diff().fillna(0)

# 4. Global Demand Features (Aggregate across EVERYTHING)
global_monthly = data.groupby("time_idx")["QTY_APPLIED"].sum().reset_index()
global_monthly = global_monthly.sort_values("time_idx")
global_monthly["global_log_qty"] = np.log1p(global_monthly["QTY_APPLIED"])
global_monthly["global_demand"] = global_monthly["global_log_qty"].shift(1) # Safe shift since it's just a time series

# 5. Merge everything back to the main dataframe
data = data.merge(supp_monthly[["SUPP_CD_HASHED", "time_idx", "avg_vol_supplier", "supplier_roll_mean6", "supplier_trend"]], on=["SUPP_CD_HASHED", "time_idx"], how="left")
data = data.merge(part_monthly[["CAT_ID_NO_HASHED", "time_idx", "avg_vol_part", "part_roll_mean6", "part_trend"]], on=["CAT_ID_NO_HASHED", "time_idx"], how="left")
data = data.merge(global_monthly[["time_idx", "global_demand"]], on="time_idx", how="left")

# Final cleanup of the new features
data.replace([np.inf, -np.inf], 0, inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)

# Keep month categorical for known categorical list
data["month"] = data["MONTHYEAR"].dt.month.astype(str).astype("category")
for col in group_ids:
    data[col] = data[col].astype(str).fillna("UNKNOWN").astype("category")

# ---------------------------
# Dataset / dataloaders
# ---------------------------
LOG.info("STEP 2: BUILD TIME-SERIES DATASET")
max_prediction_length = 6
max_encoder_length = 36
training_cutoff = data["time_idx"].max() - max_prediction_length

categorical_encoders = {
    "SUPP_CD_HASHED": NaNLabelEncoder(add_nan=True),
    "CAT_ID_NO_HASHED": NaNLabelEncoder(add_nan=True),
    "month": NaNLabelEncoder(add_nan=True),
}

training = TimeSeriesDataSet(
    data[data.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="log_ship_qty",
    group_ids=group_ids,
    min_encoder_length=3,
    min_prediction_length=1,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=group_ids,
    time_varying_known_categoricals=["month"],
    time_varying_known_reals=[
        "time_idx",
        "log_ordered_qty",
        "log_order_lag1", 
        "log_order_roll3",
    ],
    time_varying_unknown_reals=[
        # target
        "log_ship_qty",
        # rolling / lag signals (past-only)
        "ship_lag1",
        "ship_lag2",
        "ship_lag3",
        "ship_roll3",
        "supplier_roll_mean6",
        "part_roll_mean6",
        "supplier_trend",
        "part_trend",
        "global_demand",
    ],
    categorical_encoders=categorical_encoders,
    target_normalizer=EncoderNormalizer(transformation=None),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# dataloader tuning — conservative workers
batch_size = 512
num_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, pin_memory=True)

# ---------------------------
# Baseline
# ---------------------------
LOG.info("STEP 3: BASELINE")
try:
    baseline_preds = Baseline().predict(val_dataloader, return_y=True)
    baseline_mae = MAE()(baseline_preds.output, baseline_preds.y)
    LOG.info(f"Baseline MAE (naive): {baseline_mae:.4f}")
except Exception as e:
    LOG.warning("Baseline failed: %s", e)

# ---------------------------
# LR-Finder with safe fallback
# ---------------------------
LOG.info("STEP 4: LR FINDER (safe)")
pl.seed_everything(42)

# create a small TFT to run LR-finder (must match important architecture choices)
tft_tune = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    optimizer="adamw",
)

tune_trainer = pl.Trainer(
    accelerator=DEVICE,
    devices=1 if DEVICE == "gpu" else None,
    limit_train_batches=100,  # fast trial
    gradient_clip_val=0.1,
    enable_checkpointing=False,
    logger=False,
)

optimal_lr = 1e-3
try:
    res = Tuner(tune_trainer).lr_find(
        tft_tune,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )
    suggested = res.suggestion()
    # ensure suggested is numeric and reasonable
    if suggested is None or not (1e-6 <= suggested <= 1.0):
        LOG.warning("LR finder suggestion invalid, using fallback 1e-3")
    else:
        optimal_lr = float(suggested)
        LOG.info(f"LR found: {optimal_lr:.6g}")
    # optionally save lr plot
    try:
        fig = res.plot(suggest=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "lr_find_plot.png"))
        plt.close(fig)
    except Exception:
        pass
except Exception as e:
    LOG.warning("LR finder failed; using fallback LR=1e-3. Error: %s", e)
    optimal_lr = 1e-3

# ---------------------------
# Training (final model)
# ---------------------------
LOG.info("STEP 5: TRAINING")

# conservative model size to reduce overfitting risk
# If you have tested and have lots of dense series, you can increase hidden_size=128 / heads=8
hidden_size = 64
attention_head_size = 4

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=MODEL_DIR,
    filename="tft-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    mode="min"
)
early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=6, mode="min")

trainer = pl.Trainer(
    accelerator=DEVICE,
    devices=1 if DEVICE == "gpu" else None,
    max_epochs=10,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback, early_stop],
    logger=TensorBoardLogger("lightning_logs"),
    enable_progress_bar=True,
    deterministic=False,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=optimal_lr,
    hidden_size=hidden_size,
    attention_head_size=attention_head_size,
    dropout=0.15,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    optimizer="adamw",
)

# train
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# ---------------------------
# Reporting & Metrics
# ---------------------------
LOG.info("STEP 6: REPORTING & METRICS")
best_ckpt = checkpoint_callback.best_model_path
if not best_ckpt:
    ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "*.ckpt")))
    best_ckpt = ckpts[-1] if ckpts else None

if not best_ckpt:
    LOG.error("No checkpoint found. Exiting.")
    raise SystemExit(1)

LOG.info("Loading best model from: %s", best_ckpt)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

# raw predictions for plotting & interpretation
raw = best_tft.predict(val_dataloader, mode="raw", return_x=True)

# plotting a few predictions
pdf = PdfPages(REPORT_PDF)
n_plots = min(20, len(raw.output))
for i in range(n_plots):
    try:
        fig = best_tft.plot_prediction(raw.x, raw.output, idx=i, add_loss_to_title=True)
        pdf.savefig(fig)
        plt.close(fig)
    except Exception:
        pass

# interpretation plot (var importance)
try:
    interpretation = best_tft.interpret_output(raw.output, reduction="sum")
    fig_int = best_tft.plot_interpretation(interpretation)
    pdf.savefig(fig_int)
    plt.close(fig_int)
except Exception:
    pass

# Compute metrics robustly (handles 2-D or 3-D outputs)
predictions = best_tft.predict(val_dataloader, return_y=True)
# predictions.output can be [N, horizon, quantiles] or [N, horizon]
y_true_raw = predictions.y
if isinstance(y_true_raw, (tuple, list)):
    y_true = y_true_raw[0]
else:
    y_true = y_true_raw

y_pred_raw = predictions.output
# handle quantiles -> pick median if present
if y_pred_raw.dim() == 3:
    q_idx = y_pred_raw.size(-1) // 2
    y_pred = y_pred_raw[:, :, q_idx]
elif y_pred_raw.dim() == 2:
    y_pred = y_pred_raw
else:
    raise RuntimeError(f"Unhandled prediction dims: {y_pred_raw.dim()}")

# If y_true is flattened (rare), reshape to match horizon
if y_true.dim() == 1 and y_pred.dim() == 2:
    horizon = y_pred.shape[1]
    if y_true.numel() % horizon == 0:
        y_true = y_true.view(-1, horizon)
    else:
        raise RuntimeError("Cannot align y_true to y_pred horizon shapes")

# Final sanity check
if y_true.shape != y_pred.shape:
    raise RuntimeError(f"Shape mismatch y_true={y_true.shape} vs y_pred={y_pred.shape}")

# Convert back to real units
y_true_real = torch.expm1(y_true).cpu()
y_pred_real = torch.expm1(y_pred).cpu()

# MAE overall & per-horizon
mae_per_horizon = torch.mean(torch.abs(y_true_real - y_pred_real), dim=0)
mae_overall = mae_per_horizon.mean().item()

# MAPE per-horizon (ignore zero actuals) and aggregated weighted
epsilon = 1e-8
mask = (y_true_real > 0)
mape_per_horizon = []
valid_counts = []
for h in range(y_true_real.shape[1]):
    m = mask[:, h]
    valid_counts.append(int(m.sum().item()))
    if m.any():
        mape_h = torch.mean(torch.abs((y_true_real[m, h] - y_pred_real[m, h]) / (y_true_real[m, h] + epsilon))) * 100.0
        mape_per_horizon.append(mape_h.item())
    else:
        mape_per_horizon.append(float("nan"))

valid_counts = np.array(valid_counts, dtype=float)
if valid_counts.sum() > 0:
    # weighted by number of non-zero observations per horizon
    mape_overall = np.nansum(np.array([ (mape_per_horizon[i] if not np.isnan(mape_per_horizon[i]) else 0.0) * valid_counts[i]
                                        for i in range(len(mape_per_horizon)) ])) / valid_counts.sum()
else:
    mape_overall = float("nan")

# WAPE
abs_err_sum = torch.sum(torch.abs(y_true_real - y_pred_real)).item()
actual_sum = torch.sum(y_true_real).item() + epsilon
wape = (abs_err_sum / actual_sum) * 100.0

LOG.info("=== Validation metrics (real units) ===")
LOG.info("MAE overall: %.4f", mae_overall)
LOG.info("MAE per horizon: %s", [float(x) for x in mae_per_horizon])
LOG.info("MAPE overall (weighted): %.2f%%", mape_overall)
LOG.info("MAPE per horizon: %s", mape_per_horizon)
LOG.info("WAPE: %.2f%%", wape)

pdf.close()
LOG.info("Report saved to %s", REPORT_PDF)
