"""
==============================================================================
IMPROVED TFT PIPELINE v2 — Supply Chain Forecasting (FIXED)
==============================================================================
FIX: Hierarchical aggregation now uses pd.NamedAgg (compatible with all
     pandas versions) instead of dict-of-tuples that caused KeyError.

FIX (Lightning mismatch):
    Use `lightning.pytorch` consistently (Trainer/LightningModule namespace
    must match what PyTorch Forecasting expects). This resolves:
    TypeError: `model` must be a `LightningModule` ... got `TemporalFusionTransformer`

Dataset: SUPP_CD_HASHED (~1500), CAT_ID_NO_HASHED, FAC_CD_HASHED (~54),
         MONTHYEAR, ORDERED_QTY, QTY_APPLIED

Usage:
    python improved_tft_pipeline_v2.py --mode train  --data your_data.csv
    python improved_tft_pipeline_v2.py --mode predict --data new_data.csv
==============================================================================
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# ✅ IMPORTANT: use lightning.pytorch consistently (not pytorch_lightning)
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")


###############################################################################
#   1. CONFIG                                                                 #
###############################################################################

class Config:
    DATA_PATH      = "./0E.csv"        # ← UPDATE
    OUTPUT_DIR     = "tft_outputs"
    CHECKPOINT_DIR = "tft_outputs/checkpoints"
    ARTIFACT_PATH  = "tft_outputs/pipeline_artifacts.pkl"

    TARGET     = "QTY_APPLIED"
    COVARIATE  = "ORDERED_QTY"
    DATE_COL   = "MONTHYEAR"
    GROUP_KEYS = ["SUPP_CD_HASHED", "CAT_ID_NO_HASHED", "FAC_CD_HASHED"]
    TIME_IDX   = "time_idx"
    SERIES_ID  = "series_id"

    MAX_ENCODER_LENGTH    = 12
    MAX_PREDICTION_LENGTH = 6

    BATCH_SIZE             = 64
    MAX_EPOCHS             = 120
    LEARNING_RATE          = 3e-4
    HIDDEN_SIZE            = 80
    ATTENTION_HEAD_SIZE    = 4
    DROPOUT                = 0.15
    HIDDEN_CONTINUOUS_SIZE = 24
    GRADIENT_CLIP          = 0.1
    PATIENCE               = 10
    NUM_WORKERS            = 0
    ACCELERATOR            = "auto"

    MIN_SERIES_LENGTH     = 18
    MIN_NONZERO_MONTHS    = 3
    OUTLIER_CLIP_QUANTILE = 0.999
    SEED = 42


def _banner(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}")


###############################################################################
#   2. LOAD & CLEAN                                                           #
###############################################################################

def load_and_clean(path: str) -> pd.DataFrame:
    _banner("STEP 1 — Load & Clean")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    df[Config.DATE_COL]  = pd.to_datetime(df[Config.DATE_COL])
    df[Config.TARGET]    = pd.to_numeric(df[Config.TARGET],    errors="coerce").fillna(0).astype(float)
    df[Config.COVARIATE] = pd.to_numeric(df[Config.COVARIATE], errors="coerce").fillna(0).astype(float)
    for c in Config.GROUP_KEYS:
        df[c] = df[c].astype(str)

    df.sort_values(Config.GROUP_KEYS + [Config.DATE_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # deduplicate
    n0 = len(df)
    df = df.groupby(Config.GROUP_KEYS + [Config.DATE_COL], as_index=False).agg(
        {Config.TARGET: "sum", Config.COVARIATE: "sum"})
    if len(df) < n0:
        print(f"  Aggregated duplicates: {n0:,} → {len(df):,}")

    df[Config.SERIES_ID] = (df[Config.GROUP_KEYS[0]] + "|" +
                            df[Config.GROUP_KEYS[1]] + "|" +
                            df[Config.GROUP_KEYS[2]])

    d_min = df[Config.DATE_COL].min()
    df[Config.TIME_IDX] = ((df[Config.DATE_COL].dt.year - d_min.year) * 12 +
                           (df[Config.DATE_COL].dt.month - d_min.month))

    clip = df[Config.TARGET].quantile(Config.OUTLIER_CLIP_QUANTILE)
    n_clip = (df[Config.TARGET] > clip).sum()
    df[Config.TARGET] = df[Config.TARGET].clip(upper=clip)
    print(f"  Clipped {n_clip:,} outliers at {clip:,.0f}")
    print(f"  Clean: {df.shape}  |  Series: {df[Config.SERIES_ID].nunique():,}")
    print(f"  Dates: {df[Config.DATE_COL].min().date()} → {df[Config.DATE_COL].max().date()}")
    return df


###############################################################################
#   3. FEATURE ENGINEERING  (80+ features, 10 categories)                     #
###############################################################################

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    _banner("STEP 2 — Feature Engineering")
    df = df.copy()
    n0 = len(df.columns)
    grp = df.groupby(Config.SERIES_ID)
    dt  = df[Config.DATE_COL].dt

    # ── 1. Calendar (9) ─────────────────────────────────────────────────
    print("  [1/10] Calendar …")
    df["month"]         = dt.month.astype(str)
    df["quarter"]       = dt.quarter.astype(str)
    df["half_year"]     = (dt.month > 6).astype(int).astype(str)
    df["is_q4"]         = (dt.quarter == 4).astype(int).astype(str)
    df["is_year_start"] = (dt.month <= 2).astype(int).astype(str)
    mnum = dt.month
    df["month_sin"]   = np.sin(2 * np.pi * mnum / 12)
    df["month_cos"]   = np.cos(2 * np.pi * mnum / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * dt.quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * dt.quarter / 4)

    # ── 2. Lags (12) ────────────────────────────────────────────────────
    print("  [2/10] Lags …")
    for lag in [1, 2, 3, 6, 12]:
        df[f"tgt_lag{lag}"] = grp[Config.TARGET].shift(lag)
        df[f"ord_lag{lag}"] = grp[Config.COVARIATE].shift(lag)
    df["tgt_yoy_diff"] = df[Config.TARGET]    - df["tgt_lag12"]
    df["ord_yoy_diff"] = df[Config.COVARIATE] - df["ord_lag12"]

    # ── 3. Rolling stats (22) ───────────────────────────────────────────
    print("  [3/10] Rolling stats …")
    for w in [3, 6, 12]:
        for stat, fn in [("mean","mean"),("std","std"),("min","min"),
                         ("max","max"),("med","median")]:
            df[f"tgt_r{stat}{w}"] = grp[Config.TARGET].transform(
                lambda x, _f=fn, _w=w: getattr(
                    x.shift(1).rolling(_w, min_periods=1), _f)())
        df[f"ord_rmean{w}"] = grp[Config.COVARIATE].transform(
            lambda x, _w=w: x.shift(1).rolling(_w, min_periods=1).mean())
    for c in [c for c in df.columns if "rstd" in c]:
        df[c] = df[c].fillna(0)
    df["tgt_ewm3"] = grp[Config.TARGET].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean())
    df["tgt_ewm6"] = grp[Config.TARGET].transform(
        lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())

    # ── 4. Fill rate (4) ────────────────────────────────────────────────
    print("  [4/10] Fill rate …")
    df["fill_rate"] = np.where(
        df[Config.COVARIATE] > 0,
        (df[Config.TARGET] / df[Config.COVARIATE]).clip(-5, 5), 0.0)
    df["fr_rmean3"] = grp["fill_rate"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["fr_rmean6"] = grp["fill_rate"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    df["fr_dev"] = (df["fill_rate"] - 1.0).abs()

    # ── 5. Trend (5) ────────────────────────────────────────────────────
    print("  [5/10] Trend …")
    df["tgt_mom"] = grp[Config.TARGET].pct_change(1).clip(-10, 10).fillna(0)
    df["momentum_3v12"] = np.where(
        df["tgt_rmean12"] > 0, df["tgt_rmean3"] / df["tgt_rmean12"], 1.0)
    df["momentum_3v6"] = np.where(
        df["tgt_rmean6"] > 0, df["tgt_rmean3"] / df["tgt_rmean6"], 1.0)
    s_now   = grp[Config.TARGET].shift(1)
    s_past  = grp[Config.TARGET].shift(6)
    df["tgt_slope6"] = np.where(s_past.notna(), (s_now - s_past) / 5.0, 0)
    df["tgt_accel"]  = grp["tgt_slope6"].diff(1).fillna(0)

    # ── 6. Hierarchical (17) — FIXED with pd.NamedAgg ──────────────────
    print("  [6/10] Hierarchical …")

    # Supplier level
    supp = (df.groupby([Config.GROUP_KEYS[0], Config.DATE_COL])
              .agg(supp_total   = pd.NamedAgg(column=Config.TARGET,    aggfunc="sum"),
                   supp_mean    = pd.NamedAgg(column=Config.TARGET,    aggfunc="mean"),
                   supp_nactive = pd.NamedAgg(column=Config.SERIES_ID, aggfunc="nunique"),
                   supp_ord     = pd.NamedAgg(column=Config.COVARIATE, aggfunc="sum"))
              .reset_index())
    df = df.merge(supp, on=[Config.GROUP_KEYS[0], Config.DATE_COL], how="left")

    # Facility level
    fac = (df.groupby([Config.GROUP_KEYS[2], Config.DATE_COL])
             .agg(fac_total   = pd.NamedAgg(column=Config.TARGET,    aggfunc="sum"),
                  fac_mean    = pd.NamedAgg(column=Config.TARGET,    aggfunc="mean"),
                  fac_nactive = pd.NamedAgg(column=Config.SERIES_ID, aggfunc="nunique"),
                  fac_ord     = pd.NamedAgg(column=Config.COVARIATE, aggfunc="sum"))
             .reset_index())
    df = df.merge(fac, on=[Config.GROUP_KEYS[2], Config.DATE_COL], how="left")

    # Catalog level
    cat = (df.groupby([Config.GROUP_KEYS[1], Config.DATE_COL])
             .agg(cat_total = pd.NamedAgg(column=Config.TARGET,        aggfunc="sum"),
                  cat_mean  = pd.NamedAgg(column=Config.TARGET,        aggfunc="mean"),
                  cat_nfac  = pd.NamedAgg(column=Config.GROUP_KEYS[2], aggfunc="nunique"))
             .reset_index())
    df = df.merge(cat, on=[Config.GROUP_KEYS[1], Config.DATE_COL], how="left")

    # Global level
    glob = (df.groupby(Config.DATE_COL)
              .agg(glob_total = pd.NamedAgg(column=Config.TARGET, aggfunc="sum"),
                   glob_mean  = pd.NamedAgg(column=Config.TARGET, aggfunc="mean"))
              .reset_index())
    df = df.merge(glob, on=Config.DATE_COL, how="left")

    # Relative shares
    df["share_supp"]   = np.where(df["supp_total"] > 0,
                                   df[Config.TARGET] / df["supp_total"], 0)
    df["share_fac"]    = np.where(df["fac_total"] > 0,
                                   df[Config.TARGET] / df["fac_total"], 0)
    df["share_global"] = np.where(df["glob_total"] > 0,
                                   df[Config.TARGET] / df["glob_total"], 0)
    df["supp_fr"] = np.where(df["supp_ord"] > 0,
                              df["supp_total"] / df["supp_ord"], 0)
    df["fac_fr"]  = np.where(df["fac_ord"] > 0,
                              df["fac_total"] / df["fac_ord"], 0)

    # ── 7. Volatility (7) ───────────────────────────────────────────────
    print("  [7/10] Volatility …")
    for w in [3, 6, 12]:
        df[f"cv{w}"] = np.where(df[f"tgt_rmean{w}"] > 0,
                                 df[f"tgt_rstd{w}"] / df[f"tgt_rmean{w}"], 0)
    df["range6"]  = df["tgt_rmax6"]  - df["tgt_rmin6"]
    df["range12"] = df["tgt_rmax12"] - df["tgt_rmin12"]
    for w in [6, 12]:
        df[f"zcnt{w}"] = grp[Config.TARGET].transform(
            lambda x, _w=w: x.shift(1).rolling(_w, min_periods=1).apply(
                lambda s: (s == 0).sum()))

    # ── 8. Demand classification (2 static cats) ────────────────────────
    print("  [8/10] Demand classification …")
    ss = (df.groupby(Config.SERIES_ID)
            .agg(_mn = pd.NamedAgg(column=Config.TARGET, aggfunc="mean"),
                 _sd = pd.NamedAgg(column=Config.TARGET, aggfunc="std"),
                 _n  = pd.NamedAgg(column=Config.TARGET, aggfunc="count"),
                 _z  = pd.NamedAgg(column=Config.TARGET,
                                   aggfunc=lambda x: (x == 0).sum()))
            .reset_index())
    ss["_sd"] = ss["_sd"].fillna(0)
    ss["_cv2"] = np.where(ss["_mn"] > 0, (ss["_sd"] / ss["_mn"])**2, 0)
    ss["_adi"] = ss["_n"] / np.maximum(ss["_n"] - ss["_z"], 1)

    ss["demand_class"] = "Smooth"
    ss.loc[(ss["_adi"] >= 1.32) & (ss["_cv2"] <  0.49), "demand_class"] = "Intermittent"
    ss.loc[(ss["_adi"] <  1.32) & (ss["_cv2"] >= 0.49), "demand_class"] = "Erratic"
    ss.loc[(ss["_adi"] >= 1.32) & (ss["_cv2"] >= 0.49), "demand_class"] = "Lumpy"

    q33, q66 = ss["_mn"].quantile([0.33, 0.66])
    ss["volume_tier"] = "Low"
    ss.loc[ss["_mn"] >= q33, "volume_tier"] = "Medium"
    ss.loc[ss["_mn"] >= q66, "volume_tier"] = "High"

    df = df.merge(ss[[Config.SERIES_ID, "demand_class", "volume_tier"]],
                  on=Config.SERIES_ID, how="left")
    for cls in ["Smooth", "Erratic", "Intermittent", "Lumpy"]:
        n = (ss["demand_class"] == cls).sum()
        print(f"    {cls:15s}: {n:>7,}  ({n/len(ss)*100:.1f}%)")

    # ── 9. Interactions (4) ─────────────────────────────────────────────
    print("  [9/10] Interactions …")
    df["ord_momentum"] = np.where(
        df["ord_rmean6"] > 0, df[Config.COVARIATE] / df["ord_rmean6"], 1.0)
    df["tgt_vs_supp"] = np.where(
        df["supp_mean"] > 0, df[Config.TARGET] / df["supp_mean"], 1.0)
    df["tgt_vs_fac"] = np.where(
        df["fac_mean"] > 0, df[Config.TARGET] / df["fac_mean"], 1.0)

    exp_mean = grp[Config.TARGET].transform(
        lambda x: x.shift(1).expanding().mean())
    exp_std  = grp[Config.TARGET].transform(
        lambda x: x.shift(1).expanding().std().fillna(1).clip(lower=0.01))
    df["tgt_zscore"] = ((df[Config.TARGET] - exp_mean) / exp_std).clip(-5, 5)

    # ── 10. Intermittency (3+1) ─────────────────────────────────────────
    print("  [10/10] Intermittency …")
    df["months_since_nz"] = grp[Config.TARGET].transform(
        lambda x: x.groupby((x != 0).cumsum()).cumcount())
    df["cum_nz"] = grp[Config.TARGET].transform(
        lambda x: (x != 0).cumsum().shift(1).fillna(0))
    df["prev_zero"] = (grp[Config.TARGET].shift(1) == 0).astype(int).astype(str)

    # ── Cleanup ──────────────────────────────────────────────────────────
    print("  Cleaning …")
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Features added: {len(df.columns) - n0}  |  Total: {len(df.columns)}")
    return df


###############################################################################
#   4. FILTER                                                                 #
###############################################################################

def filter_series(df):
    _banner("STEP 3 — Filter")
    n_s0, n_r0 = df[Config.SERIES_ID].nunique(), len(df)

    lens = df.groupby(Config.SERIES_ID)[Config.TIME_IDX].transform("count")
    df = df[lens >= Config.MIN_SERIES_LENGTH]
    nz = df.groupby(Config.SERIES_ID)[Config.TARGET].transform(lambda x: (x != 0).sum())
    df = df[nz >= Config.MIN_NONZERO_MONTHS].reset_index(drop=True)

    print(f"  Series: {n_s0:,} → {df[Config.SERIES_ID].nunique():,}")
    print(f"  Rows:   {n_r0:,} → {len(df):,}")
    return df


###############################################################################
#   5. FEATURE REGISTRY                                                       #
###############################################################################

def build_registry(df) -> Dict:
    _banner("STEP 4 — Feature Registry")
    reg = dict(
        static_categoricals = Config.GROUP_KEYS + ["demand_class", "volume_tier"],
        time_varying_known_categoricals = [
            "month", "quarter", "half_year", "is_q4", "is_year_start"],
        time_varying_known_reals = [
            Config.TIME_IDX, Config.COVARIATE,
            "month_sin", "month_cos", "quarter_sin", "quarter_cos"],
        time_varying_unknown_categoricals = ["prev_zero"],
        time_varying_unknown_reals = [Config.TARGET]
            + [f"tgt_lag{l}" for l in [1,2,3,6,12]]
            + [f"ord_lag{l}" for l in [1,2,3,6,12]]
            + ["tgt_yoy_diff", "ord_yoy_diff"]
            + [f"tgt_r{s}{w}" for w in [3,6,12]
                               for s in ["mean","std","min","max","med"]]
            + [f"ord_rmean{w}" for w in [3,6,12]]
            + ["tgt_ewm3", "tgt_ewm6"]
            + ["fill_rate", "fr_rmean3", "fr_rmean6", "fr_dev"]
            + ["tgt_mom", "momentum_3v12", "momentum_3v6",
               "tgt_slope6", "tgt_accel"]
            + ["supp_total", "supp_mean", "supp_nactive",
               "fac_total",  "fac_mean",  "fac_nactive",
               "cat_total",  "cat_mean",  "cat_nfac",
               "glob_total", "glob_mean",
               "share_supp", "share_fac", "share_global",
               "supp_fr",    "fac_fr"]
            + [f"cv{w}" for w in [3,6,12]]
            + ["range6", "range12", "zcnt6", "zcnt12"]
            + ["ord_momentum", "tgt_vs_supp", "tgt_vs_fac", "tgt_zscore"]
            + ["months_since_nz", "cum_nz"],
    )
    # validate
    all_f = sum(reg.values(), [])
    missing = [f for f in all_f if f not in df.columns]
    if missing:
        print(f"  ⚠ Missing (removing): {missing}")
        for m in missing:
            for v in reg.values():
                if m in v:
                    v.remove(m)

    total = sum(len(v) for v in reg.values())
    for k, v in reg.items():
        print(f"  {k:45s}: {len(v):>3}")
    print(f"  {'TOTAL':45s}: {total:>3}")
    return reg


###############################################################################
#   6–7. SPLIT + DATASETS                                                     #
###############################################################################

def split(df):
    _banner("STEP 5 — Split")
    cutoff = df[Config.TIME_IDX].max() - Config.MAX_PREDICTION_LENGTH
    train = df[df[Config.TIME_IDX] <= cutoff].copy()
    print(f"  Cutoff={cutoff}  Train={len(train):,}  Full={len(df):,}")
    return train, df.copy(), cutoff


def build_datasets(train_df, full_df, reg):
    _banner("STEP 6 — Datasets")
    training = TimeSeriesDataSet(
        train_df,
        time_idx=Config.TIME_IDX, target=Config.TARGET,
        group_ids=Config.GROUP_KEYS,
        max_encoder_length=Config.MAX_ENCODER_LENGTH,
        max_prediction_length=Config.MAX_PREDICTION_LENGTH,
        static_categoricals=reg["static_categoricals"],
        time_varying_known_categoricals=reg["time_varying_known_categoricals"],
        time_varying_known_reals=reg["time_varying_known_reals"],
        time_varying_unknown_categoricals=reg["time_varying_unknown_categoricals"],
        time_varying_unknown_reals=reg["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(
            groups=Config.GROUP_KEYS, transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True,
        add_encoder_length=True, allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, full_df, predict=True, stop_randomization=True)
    print(f"  Train samples: {len(training):,}  Val samples: {len(validation):,}")

    train_dl = training.to_dataloader(
        train=True, batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS, persistent_workers=False)
    val_dl = validation.to_dataloader(
        train=False, batch_size=Config.BATCH_SIZE*2,
        num_workers=Config.NUM_WORKERS, persistent_workers=False)
    return training, validation, train_dl, val_dl


###############################################################################
#   8–9. MODEL + TRAIN                                                        #
###############################################################################

def create_model(training):
    _banner("STEP 7 — Model")
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=Config.LEARNING_RATE, hidden_size=Config.HIDDEN_SIZE,
        attention_head_size=Config.ATTENTION_HEAD_SIZE, dropout=Config.DROPOUT,
        hidden_continuous_size=Config.HIDDEN_CONTINUOUS_SIZE,
        output_size=7, loss=QuantileLoss(),
        log_interval=10, reduce_on_plateau_patience=5)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n:,}  hidden={Config.HIDDEN_SIZE}")
    return model


def train_model(model, train_dl, val_dl):
    _banner("STEP 8 — Training")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR, filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1, verbose=True)
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS, accelerator=Config.ACCELERATOR,
        gradient_clip_val=Config.GRADIENT_CLIP,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=Config.PATIENCE,
                          verbose=True, mode="min"),
            LearningRateMonitor(logging_interval="epoch"), ckpt_cb],
        logger=TensorBoardLogger(Config.OUTPUT_DIR, name="logs"),
        enable_progress_bar=True, log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"  Best: {ckpt_cb.best_model_path}  loss={ckpt_cb.best_model_score:.4f}")
    return trainer, ckpt_cb.best_model_path


###############################################################################
#   10–11. SAVE + EVALUATE                                                    #
###############################################################################

def save_artifacts(training_ds, best_ckpt, registry):
    _banner("STEP 9 — Save")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(Config.ARTIFACT_PATH, "wb") as f:
        pickle.dump({"dataset_params": training_ds.get_parameters(),
                      "registry": registry, "best_ckpt": best_ckpt}, f)
    print(f"  Saved: {Config.ARTIFACT_PATH}")


def load_artifacts():
    with open(Config.ARTIFACT_PATH, "rb") as f:
        blob = pickle.load(f)
    model = TemporalFusionTransformer.load_from_checkpoint(blob["best_ckpt"])
    model.eval()
    return model, blob


def evaluate(model, val_dl):
    _banner("STEP 10 — Evaluate")

    # Ensure model device
    device = next(model.parameters()).device

    # Get raw predictions (this will follow model/device)
    raw = model.predict(val_dl, mode="raw", return_x=True)

    # Median quantile (index 3 in your setup)
    pred = raw.output["prediction"][:, :, 3]

    # Collect actuals from val_dl and move to same device as pred
    # y from TimeSeriesDataSet dataloader is typically a tuple/list; y[0] is target
    acts = []
    for x, y in iter(val_dl):
        # y can be (target, weight) or similar; handle both
        y0 = y[0] if isinstance(y, (tuple, list)) else y
        acts.append(y0)

    act = torch.cat(acts, dim=0)

    # Align device + dtype
    pred = pred.to(device)
    act = act.to(device)

    # Safety: trim to same batch dimension if needed
    n = min(pred.shape[0], act.shape[0])
    pred, act = pred[:n], act[:n]

    # Metrics
    mae  = torch.mean(torch.abs(pred - act)).item()
    rmse = torch.sqrt(torch.mean((pred - act) ** 2)).item()

    nz = act != 0
    mape = (torch.abs((act[nz] - pred[nz]) / act[nz])).mean().item() * 100 if nz.any() else float("nan")

    denom = (torch.abs(act) + torch.abs(pred)) / 2
    dm = denom > 0
    smape = (torch.abs(act[dm] - pred[dm]) / denom[dm]).mean().item() * 100 if dm.any() else float("nan")

    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  sMAPE={smape:.2f}%")
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, sMAPE=smape)


###############################################################################
#   12. VISUALIZE                                                             #
###############################################################################

def visualize(model, val_dl):
    _banner("STEP 11 — Visualize")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    raw = model.predict(val_dl, mode="raw", return_x=True)

    n_plot = min(8, raw.output["prediction"].shape[0])
    fig, axes = plt.subplots(n_plot, 1, figsize=(16, 3.5*n_plot))
    if n_plot == 1:
        axes = [axes]
    h = range(1, Config.MAX_PREDICTION_LENGTH+1)
    for i, ax in enumerate(axes):
        p = raw.output["prediction"][i].detach().cpu().numpy()
        ax.plot(h, p[:,3], "b-o", ms=4, label="Median")
        ax.fill_between(h, p[:,1], p[:,5], alpha=0.22, color="blue", label="80% CI")
        ax.fill_between(h, p[:,0], p[:,6], alpha=0.08, color="blue", label="96% CI")
        ax.set_title(f"Series #{i}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/forecast_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: forecast_samples.png")

    try:
        interp = model.interpret_output(raw.output, reduction="sum")
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        for ax, key, title, color, top_n in [
            (axes[0], "encoder_variables", "Encoder (top 25)", "#2196F3", 25),
            (axes[1], "decoder_variables", "Decoder (top 15)", "#FF9800", 15),
        ]:
            srt = dict(sorted(interp[key].items(), key=lambda x: x[1], reverse=True)[:top_n])
            ax.barh(range(len(srt)), list(srt.values()), color=color, alpha=0.85)
            ax.set_yticks(range(len(srt)))
            ax.set_yticklabels(list(srt.keys()), fontsize=7)
            ax.invert_yaxis()
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_DIR}/variable_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: variable_importance.png")
    except Exception as e:
        print(f"  Var importance skipped: {e}")


###############################################################################
#   13. INFERENCE                                                             #
###############################################################################

def run_inference(data_path):
    _banner("INFERENCE")
    model, blob = load_artifacts()
    df = load_and_clean(data_path)
    df = engineer_features(df)
    df = filter_series(df)
    pred_ds = TimeSeriesDataSet.from_parameters(
        blob["dataset_params"], df, predict=True, stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(
        train=False, batch_size=Config.BATCH_SIZE*2, num_workers=Config.NUM_WORKERS)
    raw = model.predict(pred_dl, mode="quantiles", return_x=True)
    idx = pred_ds.x_to_index(raw.x)

    rows = []
    qn = ["q02","q10","q25","q50","q75","q90","q98"]
    for i in range(len(idx)):
        r = idx.iloc[i]
        sid = "|".join(str(r[g]) for g in Config.GROUP_KEYS)
        for h in range(Config.MAX_PREDICTION_LENGTH):
            rec = {"series_id": sid, "horizon": h+1,
                   "predicted": raw.output[i,h,3].item()}
            for qi, q in enumerate(qn):
                rec[q] = raw.output[i,h,qi].item()
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(f"{Config.OUTPUT_DIR}/predictions.csv", index=False)
    print(f"  {len(out):,} rows  |  {out['series_id'].nunique():,} series")
    return out


###############################################################################
#   14. MAIN                                                                  #
###############################################################################

def run_full_pipeline(data_path):
    pl.seed_everything(Config.SEED, workers=True)
    df  = load_and_clean(data_path)
    df  = engineer_features(df)
    df  = filter_series(df)
    reg = build_registry(df)
    train_df, full_df, _ = split(df)
    training_ds, _, train_dl, val_dl = build_datasets(train_df, full_df, reg)
    model = create_model(training_ds)
    _, best_ckpt = train_model(model, train_dl, val_dl)
    best = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    metrics = evaluate(best, val_dl)
    visualize(best, val_dl)
    save_artifacts(training_ds, best_ckpt, reg)
    print(f"\n{'█'*70}\n  DONE — MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
          f"sMAPE={metrics['sMAPE']:.2f}%\n{'█'*70}\n")
    return best, training_ds, metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","predict"], default="train")
    ap.add_argument("--data", type=str, default=None)
    a = ap.parse_args()
    if a.data:
        Config.DATA_PATH = a.data
    if a.mode == "train":
        run_full_pipeline(Config.DATA_PATH)
    else:
        run_inference(Config.DATA_PATH)
