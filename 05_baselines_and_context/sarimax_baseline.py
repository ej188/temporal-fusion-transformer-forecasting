#!/usr/bin/env python3
"""
Consolidated SARIMAX baseline for supply-chain shipment forecasting.

This file combines the useful pieces from:
- mj_SARIMAX.py
- sarimax_supply_chain_forecast (4).py
- sarimax_supply_chain_forecast (5).py

It is intended as a public baseline companion to the TFT implementation. It does
not hard-code private paths and can run in two modes:

1. volume
   Model the highest-volume supplier-part series.
2. fill_rate
   Rank suppliers by fulfillment/fill-rate quality, then model the top supplier
   subset. This preserves the project-impact idea from the fill-rate script.

Expected input columns:
- SUPP_CD_HASHED
- CAT_ID_NO_HASHED
- FAC_CD_HASHED
- MONTHYEAR
- ORDERED_QTY
- QTY_APPLIED
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


REQUIRED_COLUMNS = [
    "SUPP_CD_HASHED",
    "CAT_ID_NO_HASHED",
    "FAC_CD_HASHED",
    "MONTHYEAR",
    "ORDERED_QTY",
    "QTY_APPLIED",
]

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)
EXOG_COLS = [
    "demand_ratio",
    "lag_12",
    "rolling_mean_3",
    "rolling_std_3",
    "month",
    "quarter",
]


@dataclass(frozen=True)
class Config:
    data_path: Path
    output_dir: Path
    mode: str
    include_facility: bool
    min_nonzero_months: int
    top_n_series: int
    test_horizon_months: int
    future_horizon_months: int
    ordered_qty_clip_q: float
    maxiter: int
    top_n_suppliers: int
    min_supplier_orders: int
    min_supplier_months: int
    log_every: int

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "sarimax_plots"

    @property
    def forecasts_csv(self) -> Path:
        return self.output_dir / "sarimax_forecasts.csv"

    @property
    def evaluation_csv(self) -> Path:
        return self.output_dir / "sarimax_evaluation.csv"

    @property
    def supplier_ranking_csv(self) -> Path:
        return self.output_dir / "supplier_fillrate_ranking.csv"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run a SARIMAX/seasonal-naive baseline for shipment forecasting."
    )
    parser.add_argument("--data", required=True, help="Path to anonymized monthly input CSV.")
    parser.add_argument("--output-dir", default="sarimax_outputs", help="Directory for CSV and plot outputs.")
    parser.add_argument(
        "--mode",
        choices=["volume", "fill_rate"],
        default="volume",
        help="Baseline selection strategy.",
    )
    parser.add_argument(
        "--include-facility",
        action="store_true",
        help="Use facility+supplier+part as the series key. Default uses supplier+part.",
    )
    parser.add_argument("--min-nonzero-months", type=int, default=24)
    parser.add_argument("--top-n-series", type=int, default=100)
    parser.add_argument("--test-horizon-months", type=int, default=6)
    parser.add_argument("--future-horizon-months", type=int, default=6)
    parser.add_argument("--ordered-qty-clip-q", type=float, default=0.995)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--top-n-suppliers", type=int, default=20)
    parser.add_argument("--min-supplier-orders", type=int, default=100)
    parser.add_argument("--min-supplier-months", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    return Config(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        include_facility=args.include_facility,
        min_nonzero_months=args.min_nonzero_months,
        top_n_series=args.top_n_series,
        test_horizon_months=args.test_horizon_months,
        future_horizon_months=args.future_horizon_months,
        ordered_qty_clip_q=args.ordered_qty_clip_q,
        maxiter=args.maxiter,
        top_n_suppliers=args.top_n_suppliers,
        min_supplier_orders=args.min_supplier_orders,
        min_supplier_months=args.min_supplier_months,
        log_every=args.log_every,
    )


def ensure_dirs(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()
    df["MONTHYEAR"] = pd.to_datetime(df["MONTHYEAR"], errors="coerce")
    bad_dates = int(df["MONTHYEAR"].isna().sum())
    if bad_dates:
        print(f"[Clean] Dropping {bad_dates:,} rows with invalid MONTHYEAR.")
    df = df.dropna(subset=["MONTHYEAR"])
    df["MONTHYEAR"] = df["MONTHYEAR"].dt.to_period("M").dt.to_timestamp()

    for col in ["ORDERED_QTY", "QTY_APPLIED"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["ORDERED_QTY"])
    dropped = before - len(df)
    if dropped:
        print(f"[Clean] Dropped {dropped:,} rows with invalid ORDERED_QTY.")

    df["QTY_APPLIED"] = df["QTY_APPLIED"].fillna(0.0)
    df["ORDERED_QTY"] = df["ORDERED_QTY"].clip(lower=0.0)
    df["QTY_APPLIED"] = df["QTY_APPLIED"].clip(lower=0.0)

    clip_val = df["ORDERED_QTY"].quantile(config.ordered_qty_clip_q)
    n_clipped = int((df["ORDERED_QTY"] > clip_val).sum())
    if n_clipped:
        print(
            f"[Clean] Clipping {n_clipped:,} ORDERED_QTY rows above "
            f"{config.ordered_qty_clip_q:.3f} quantile ({clip_val:.2f})."
        )
        df.loc[df["ORDERED_QTY"] > clip_val, "ORDERED_QTY"] = clip_val

    return df


def aggregate_duplicates(df: pd.DataFrame, include_facility: bool) -> pd.DataFrame:
    key_cols = series_key_columns(include_facility) + ["MONTHYEAR"]
    dup_count = int(df.duplicated(subset=key_cols).sum())
    if dup_count:
        print(f"[Validate] Aggregating {dup_count:,} duplicate monthly series rows.")
        df = (
            df.groupby(key_cols, as_index=False)
            .agg({"ORDERED_QTY": "sum", "QTY_APPLIED": "sum"})
        )
    else:
        print("[Validate] No duplicate monthly series rows found.")
    return df


def series_key_columns(include_facility: bool) -> list[str]:
    if include_facility:
        return ["FAC_CD_HASHED", "SUPP_CD_HASHED", "CAT_ID_NO_HASHED"]
    return ["SUPP_CD_HASHED", "CAT_ID_NO_HASHED"]


def add_unique_id(df: pd.DataFrame, include_facility: bool) -> pd.DataFrame:
    df = df.copy()
    key_cols = series_key_columns(include_facility)
    df["unique_id"] = df[key_cols].astype(str).agg("*".join, axis=1)
    df["supplier_id"] = df["SUPP_CD_HASHED"].astype(str)
    return df


def reindex_series_monthly(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for uid, group in df.groupby("unique_id", sort=False):
        group = group.sort_values("MONTHYEAR").set_index("MONTHYEAR")
        idx = pd.date_range(group.index.min(), group.index.max(), freq="MS")
        group = group.reindex(idx)
        group.index.name = "MONTHYEAR"
        group["unique_id"] = uid
        group["supplier_id"] = str(group["supplier_id"].dropna().iloc[0]) if group["supplier_id"].notna().any() else uid.split("*")[0]
        group["ORDERED_QTY"] = group["ORDERED_QTY"].fillna(0.0)
        group["QTY_APPLIED"] = group["QTY_APPLIED"].fillna(0.0)
        frames.append(group.reset_index())

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[Series] Built {out['unique_id'].nunique():,} monthly series.")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["unique_id", "MONTHYEAR"]).copy()
    df["demand_ratio"] = np.where(
        df["ORDERED_QTY"] > 0,
        df["QTY_APPLIED"] / df["ORDERED_QTY"],
        0.0,
    ).clip(0.0, 2.0)

    grouped_target = df.groupby("unique_id")["ORDERED_QTY"]
    df["lag_12"] = grouped_target.shift(12)
    df["rolling_mean_3"] = grouped_target.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_std_3"] = grouped_target.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).std().fillna(0.0)
    )
    df["month"] = df["MONTHYEAR"].dt.month.astype(int)
    df["quarter"] = df["MONTHYEAR"].dt.quarter.astype(int)

    for col in EXOG_COLS:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def rank_suppliers_by_fillrate(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df["ORDERED_QTY"] > 0].copy()
    active["fill_rate"] = (active["QTY_APPLIED"] / active["ORDERED_QTY"]).clip(0.0, 2.0)

    ranking = (
        active.groupby("SUPP_CD_HASHED")
        .agg(
            fill_rate_mean=("fill_rate", "mean"),
            fill_rate_median=("fill_rate", "median"),
            fill_rate_std=("fill_rate", "std"),
            total_ordered=("ORDERED_QTY", "sum"),
            total_applied=("QTY_APPLIED", "sum"),
            active_months=("MONTHYEAR", "nunique"),
            n_items=("CAT_ID_NO_HASHED", "nunique"),
            n_orders=("ORDERED_QTY", "count"),
        )
        .reset_index()
    )

    ranking["fill_rate_std"] = ranking["fill_rate_std"].fillna(0.0)
    ranking["overall_fill"] = np.where(
        ranking["total_ordered"] > 0,
        ranking["total_applied"] / ranking["total_ordered"],
        0.0,
    ).clip(0.0, 2.0)

    ranking["fill_closeness"] = 1.0 - np.abs(ranking["fill_rate_mean"] - 1.0)
    ranking["consistency"] = 1.0 - ranking["fill_rate_std"].clip(0.0, 1.0)
    volume_rank = ranking["total_ordered"].rank(pct=True)
    ranking["composite_score"] = (
        0.50 * ranking["fill_closeness"]
        + 0.30 * ranking["consistency"]
        + 0.20 * volume_rank
    )

    ranking = ranking.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def filter_to_top_suppliers(df: pd.DataFrame, ranking: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    qualified = ranking[
        (ranking["n_orders"] >= config.min_supplier_orders)
        & (ranking["active_months"] >= config.min_supplier_months)
    ].copy()
    top_suppliers = qualified.head(config.top_n_suppliers).copy()
    print(
        f"[Suppliers] Qualified {len(qualified):,} / {len(ranking):,}; "
        f"selected top {len(top_suppliers):,}."
    )

    selected_ids = set(top_suppliers["SUPP_CD_HASHED"].astype(str))
    filtered = df[df["SUPP_CD_HASHED"].astype(str).isin(selected_ids)].copy()
    return filtered, top_suppliers


def select_series(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = (
        df.groupby("unique_id")
        .agg(
            supplier_id=("supplier_id", "first"),
            total_demand=("ORDERED_QTY", "sum"),
            nonzero_months=("ORDERED_QTY", lambda s: int((s > 0).sum())),
            n_months=("ORDERED_QTY", "size"),
        )
        .reset_index()
    )

    eligible = stats[stats["nonzero_months"] >= config.min_nonzero_months].copy()
    eligible = eligible.sort_values("total_demand", ascending=False)
    selected = eligible.head(config.top_n_series).copy()
    print(
        f"[Select] Eligible series: {len(eligible):,} / {len(stats):,}; "
        f"selected top {len(selected):,}."
    )

    selected_ids = set(selected["unique_id"])
    return df[df["unique_id"].isin(selected_ids)].copy(), selected


def train_test_split_series(group: pd.DataFrame, test_horizon_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values("MONTHYEAR").copy()
    cutoff = group["MONTHYEAR"].max() - pd.DateOffset(months=test_horizon_months - 1)
    train = group[group["MONTHYEAR"] < cutoff].copy()
    test = group[group["MONTHYEAR"] >= cutoff].copy()
    return train, test


def safe_mape(actual: Iterable[float], pred: Iterable[float]) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return np.nan
    return float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100.0)


def wmape_score(actual: Iterable[float], pred: Iterable[float]) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom * 100.0)


def seasonal_naive_forecast(train_y: Iterable[float], steps: int, seasonal_period: int = 12) -> np.ndarray:
    y = np.asarray(train_y, dtype=float)
    if len(y) == 0:
        return np.zeros(steps, dtype=float)

    forecast = []
    for i in range(steps):
        idx = len(y) + i - seasonal_period
        if 0 <= idx < len(y):
            forecast.append(y[idx])
        else:
            forecast.append(y[-1])
    return np.asarray(forecast, dtype=float)


def build_future_exog(group: pd.DataFrame, future_idx: pd.DatetimeIndex) -> pd.DataFrame:
    recent = group.tail(3)
    recent_target_mean = float(recent["ORDERED_QTY"].mean()) if len(recent) else 0.0
    recent_target_std = float(recent["ORDERED_QTY"].std()) if len(recent) > 1 else 0.0
    recent_ratio = float(recent["demand_ratio"].mean()) if len(recent) else 0.0

    return pd.DataFrame(
        {
            "demand_ratio": np.full(len(future_idx), recent_ratio, dtype=float),
            "lag_12": np.full(len(future_idx), recent_target_mean, dtype=float),
            "rolling_mean_3": np.full(len(future_idx), recent_target_mean, dtype=float),
            "rolling_std_3": np.full(len(future_idx), recent_target_std, dtype=float),
            "month": future_idx.month.astype(int),
            "quarter": ((future_idx.month - 1) // 3 + 1).astype(int),
        }
    )


def result_template(group: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    uid = group["unique_id"].iloc[0]
    supplier_id = group["supplier_id"].iloc[0]
    return {
        "unique_id": uid,
        "supplier_id": supplier_id,
        "train_start": train["MONTHYEAR"].min() if not train.empty else pd.NaT,
        "train_end": train["MONTHYEAR"].max() if not train.empty else pd.NaT,
        "test_start": test["MONTHYEAR"].min() if not test.empty else pd.NaT,
        "test_end": test["MONTHYEAR"].max() if not test.empty else pd.NaT,
        "model": None,
        "mae_test": np.nan,
        "mape_test": np.nan,
        "wmape_test": np.nan,
        "fitted_values": None,
        "fitted_index": None,
        "test_forecast": None,
        "test_index": test["MONTHYEAR"].values,
        "y_test": test["ORDERED_QTY"].astype(float).values,
        "future_forecast": None,
        "future_index": None,
        "error": None,
    }


def fit_and_forecast_one_series(group: pd.DataFrame, config: Config) -> dict:
    train, test = train_test_split_series(group, config.test_horizon_months)
    result = result_template(group, train, test)

    if train.empty or test.empty:
        result.update({"model": "Failed", "error": "Empty train or test split"})
        return result

    y_train = train["ORDERED_QTY"].astype(float).values
    y_test = test["ORDERED_QTY"].astype(float).values
    x_train = train[EXOG_COLS].astype(float)
    x_test = test[EXOG_COLS].astype(float)

    try:
        model = SARIMAX(
            endog=y_train,
            exog=x_train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model.fit(disp=False, maxiter=config.maxiter)
        fitted = np.asarray(model_fit.fittedvalues, dtype=float)
        fitted_idx = train["MONTHYEAR"].iloc[-len(fitted) :].values

        test_forecast = np.maximum(
            np.asarray(model_fit.forecast(steps=len(test), exog=x_test), dtype=float),
            0.0,
        )

        future_idx = pd.date_range(
            group["MONTHYEAR"].max() + pd.offsets.MonthBegin(1),
            periods=config.future_horizon_months,
            freq="MS",
        )
        future_forecast = np.maximum(
            np.asarray(
                model_fit.forecast(steps=len(future_idx), exog=build_future_exog(group, future_idx)),
                dtype=float,
            ),
            0.0,
        )

        result.update(
            {
                "model": "SARIMAX",
                "mae_test": float(mean_absolute_error(y_test, test_forecast)),
                "mape_test": safe_mape(y_test, test_forecast),
                "wmape_test": wmape_score(y_test, test_forecast),
                "fitted_values": fitted,
                "fitted_index": fitted_idx,
                "test_forecast": test_forecast,
                "future_forecast": future_forecast,
                "future_index": future_idx.values,
            }
        )
        return result
    except Exception as exc:
        return seasonal_naive_result(group, train, test, result, config, exc)


def seasonal_naive_result(
    group: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    result: dict,
    config: Config,
    sarimax_error: Exception,
) -> dict:
    try:
        y_train = train["ORDERED_QTY"].astype(float).values
        y_test = test["ORDERED_QTY"].astype(float).values
        test_forecast = np.maximum(seasonal_naive_forecast(y_train, len(test)), 0.0)
        future_forecast = np.maximum(
            seasonal_naive_forecast(np.concatenate([y_train, y_test]), config.future_horizon_months),
            0.0,
        )
        future_idx = pd.date_range(
            group["MONTHYEAR"].max() + pd.offsets.MonthBegin(1),
            periods=config.future_horizon_months,
            freq="MS",
        )
        result.update(
            {
                "model": "SeasonalNaive",
                "mae_test": float(mean_absolute_error(y_test, test_forecast)),
                "mape_test": safe_mape(y_test, test_forecast),
                "wmape_test": wmape_score(y_test, test_forecast),
                "test_forecast": test_forecast,
                "future_forecast": future_forecast,
                "future_index": future_idx.values,
                "error": f"SARIMAX failed: {repr(sarimax_error)}",
            }
        )
    except Exception as naive_error:
        result.update(
            {
                "model": "Failed",
                "error": f"SARIMAX failed: {repr(sarimax_error)} | SeasonalNaive failed: {repr(naive_error)}",
            }
        )
    return result


def summarize_metrics(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        if result["model"] not in ("SARIMAX", "SeasonalNaive"):
            continue
        rows.append(
            {
                "unique_id": result["unique_id"],
                "supplier_id": result["supplier_id"],
                "MAE_Test": result["mae_test"],
                "MAPE_Test": result["mape_test"],
                "WMAPE_Test": result["wmape_test"],
                "Model": result["model"],
                "Error": result["error"],
            }
        )
    return pd.DataFrame(rows)


def print_metric_summary(eval_df: pd.DataFrame, results: list[dict]) -> None:
    if eval_df.empty:
        print("[Eval] No successful models to summarize.")
        return

    for metric in ["MAE_Test", "MAPE_Test", "WMAPE_Test"]:
        series = eval_df[metric]
        print(
            f"[Eval] {metric}: mean={np.nanmean(series):.2f}, "
            f"median={np.nanmedian(series):.2f}, min={np.nanmin(series):.2f}, max={np.nanmax(series):.2f}"
        )

    total_actual = 0.0
    total_abs_error = 0.0
    for result in results:
        if result["model"] not in ("SARIMAX", "SeasonalNaive"):
            continue
        if result["y_test"] is None or result["test_forecast"] is None:
            continue
        total_actual += float(np.sum(np.abs(result["y_test"])))
        total_abs_error += float(np.sum(np.abs(result["y_test"] - result["test_forecast"])))

    overall_wmape = total_abs_error / total_actual * 100.0 if total_actual > 0 else np.nan
    print(f"[Eval] Overall WMAPE across modeled series: {overall_wmape:.2f}%")

    supplier_perf = (
        eval_df.groupby("supplier_id")
        .agg(
            n_series=("unique_id", "count"),
            mean_MAE=("MAE_Test", "mean"),
            mean_MAPE=("MAPE_Test", lambda s: np.nanmean(s)),
            mean_WMAPE=("WMAPE_Test", lambda s: np.nanmean(s)),
        )
        .sort_values("mean_MAE")
        .reset_index()
    )
    print("[Eval] Supplier-level baseline summary:")
    print(supplier_perf.head(20).to_string(index=False))


def plot_supplier_ranking(top_suppliers: pd.DataFrame, outpath: Path) -> None:
    if top_suppliers.empty:
        return
    data = top_suppliers.head(20).sort_values("composite_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(data["SUPP_CD_HASHED"].astype(str), data["composite_score"], color="steelblue")
    ax.set_xlabel("Composite fill-rate score")
    ax.set_title("Selected suppliers by fill-rate score")
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_fillrate_distribution(ranking: pd.DataFrame, outpath: Path) -> None:
    if ranking.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(ranking["fill_rate_mean"].clip(0, 1.5), bins=40, ax=axes[0], color="steelblue")
    axes[0].axvline(1.0, color="red", linestyle="--", label="perfect fill rate")
    axes[0].set_title("Supplier fill-rate distribution")
    axes[0].set_xlabel("Mean fill rate")
    axes[0].legend()

    sns.histplot(ranking["fill_rate_std"].clip(0, 1.0), bins=40, ax=axes[1], color="coral")
    axes[1].set_title("Supplier fill-rate variability")
    axes[1].set_xlabel("Fill-rate standard deviation")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def safe_plot_name(unique_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in unique_id)


def plot_fitted_vs_actual(train_df: pd.DataFrame, result: dict, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df["MONTHYEAR"], train_df["ORDERED_QTY"], label="actual train")
    if result["model"] == "SARIMAX" and result["fitted_values"] is not None:
        ax.plot(result["fitted_index"], result["fitted_values"], linestyle="--", label="fitted train")
    ax.set_title(f"SARIMAX fitted vs actual: {result['unique_id']}")
    ax.set_xlabel("Month")
    ax.set_ylabel("ORDERED_QTY")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_forecast_vs_actual(train_df: pd.DataFrame, test_df: pd.DataFrame, result: dict, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df["MONTHYEAR"], train_df["ORDERED_QTY"], alpha=0.45, label="actual train")
    ax.plot(test_df["MONTHYEAR"], test_df["ORDERED_QTY"], linewidth=2, label="actual test")
    ax.plot(test_df["MONTHYEAR"], result["test_forecast"], linestyle="--", marker="s", label="test forecast")
    if result["future_forecast"] is not None:
        ax.plot(result["future_index"], result["future_forecast"], linestyle=":", marker="^", label="future forecast")
    mape = result["mape_test"]
    mape_label = f"{mape:.1f}%" if np.isfinite(mape) else "N/A"
    ax.set_title(f"Forecast vs actual: {result['unique_id']} | MAE={result['mae_test']:.2f} | MAPE={mape_label}")
    ax.set_xlabel("Month")
    ax.set_ylabel("ORDERED_QTY")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_residual_diagnostics(train_df: pd.DataFrame, result: dict, outpath: Path) -> None:
    if result["model"] != "SARIMAX" or result["fitted_values"] is None:
        return
    fitted_idx = pd.to_datetime(result["fitted_index"])
    actual = train_df.set_index("MONTHYEAR").loc[fitted_idx, "ORDERED_QTY"].values
    residuals = actual - result["fitted_values"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(fitted_idx, residuals)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Residuals over time")
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title("Residual histogram")
    lags = min(24, len(residuals) - 1) if len(residuals) > 2 else 1
    plot_acf(residuals, ax=axes[2], lags=lags)
    axes[2].set_title("Residual ACF")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_error_distributions(eval_df: pd.DataFrame, outpath: Path) -> None:
    if eval_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(eval_df["MAE_Test"].dropna(), ax=axes[0])
    axes[0].set_title("MAE distribution")
    axes[0].set_xlabel("MAE")
    sns.histplot(eval_df["MAPE_Test"].dropna(), ax=axes[1])
    axes[1].set_title("MAPE distribution")
    axes[1].set_xlabel("MAPE (%)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_top_series(selected_stats: pd.DataFrame, outpath: Path) -> None:
    if selected_stats.empty:
        return
    data = selected_stats.sort_values("total_demand", ascending=False).head(10)
    data = data.sort_values("total_demand", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["unique_id"], data["total_demand"])
    ax.set_title("Top selected series by total ordered quantity")
    ax.set_xlabel("Total ORDERED_QTY")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def generate_plots(
    df_selected: pd.DataFrame,
    selected_stats: pd.DataFrame,
    results: list[dict],
    eval_df: pd.DataFrame,
    config: Config,
    supplier_ranking: pd.DataFrame | None = None,
    top_suppliers: pd.DataFrame | None = None,
) -> None:
    if supplier_ranking is not None and top_suppliers is not None:
        plot_supplier_ranking(top_suppliers, config.plots_dir / "supplier_ranking.png")
        plot_fillrate_distribution(supplier_ranking, config.plots_dir / "supplier_fillrate_distribution.png")

    top_ids = selected_stats.sort_values("total_demand", ascending=False).head(3)["unique_id"].tolist()
    result_by_id = {result["unique_id"]: result for result in results}
    for uid in top_ids:
        group = df_selected[df_selected["unique_id"] == uid].sort_values("MONTHYEAR")
        train, test = train_test_split_series(group, config.test_horizon_months)
        result = result_by_id[uid]
        name = safe_plot_name(uid)
        plot_fitted_vs_actual(train, result, config.plots_dir / f"fitted_vs_actual_{name}.png")
        plot_forecast_vs_actual(train, test, result, config.plots_dir / f"forecast_vs_actual_{name}.png")

    if top_ids:
        uid = top_ids[0]
        group = df_selected[df_selected["unique_id"] == uid].sort_values("MONTHYEAR")
        train, _ = train_test_split_series(group, config.test_horizon_months)
        plot_residual_diagnostics(train, result_by_id[uid], config.plots_dir / f"residual_diagnostics_{safe_plot_name(uid)}.png")

    plot_error_distributions(eval_df, config.plots_dir / "error_distributions.png")
    plot_top_series(selected_stats, config.plots_dir / "top_selected_series.png")


def export_results(results: list[dict], eval_df: pd.DataFrame, config: Config) -> None:
    forecast_rows = []
    for result in results:
        if result["model"] not in ("SARIMAX", "SeasonalNaive"):
            continue
        if result["future_forecast"] is None or result["future_index"] is None:
            continue
        for date, value in zip(pd.to_datetime(result["future_index"]), result["future_forecast"]):
            forecast_rows.append(
                {
                    "unique_id": result["unique_id"],
                    "supplier_id": result["supplier_id"],
                    "forecast_month": date.strftime("%Y-%m-%d"),
                    "forecast_value": round(float(value), 2),
                    "model": result["model"],
                }
            )

    pd.DataFrame(forecast_rows).to_csv(config.forecasts_csv, index=False)
    eval_df.to_csv(config.evaluation_csv, index=False)
    print(f"[Export] Forecasts: {config.forecasts_csv} ({len(forecast_rows):,} rows)")
    print(f"[Export] Evaluation: {config.evaluation_csv} ({len(eval_df):,} rows)")


def prepare_dataset(config: Config) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    raw = load_data(config.data_path)
    print(f"[Load] Loaded {len(raw):,} rows and {len(raw.columns):,} columns.")
    print(f"[Load] Facilities detected: {raw['FAC_CD_HASHED'].nunique():,}")

    cleaned = clean_data(raw, config)
    aggregated = aggregate_duplicates(cleaned, include_facility=config.include_facility)

    supplier_ranking = None
    top_suppliers = None
    if config.mode == "fill_rate":
        supplier_ranking = rank_suppliers_by_fillrate(aggregated)
        supplier_ranking.to_csv(config.supplier_ranking_csv, index=False)
        aggregated, top_suppliers = filter_to_top_suppliers(aggregated, supplier_ranking, config)
        print(f"[Export] Supplier ranking: {config.supplier_ranking_csv}")

    series_df = add_unique_id(aggregated, config.include_facility)
    series_df = reindex_series_monthly(series_df)
    series_df = add_features(series_df)
    return series_df, supplier_ranking, top_suppliers


def fit_all_series(df_selected: pd.DataFrame, selected_stats: pd.DataFrame, config: Config) -> list[dict]:
    results = []
    unique_ids = selected_stats["unique_id"].tolist()
    for idx, uid in enumerate(unique_ids, start=1):
        group = df_selected[df_selected["unique_id"] == uid].sort_values("MONTHYEAR").copy()
        result = fit_and_forecast_one_series(group, config)
        if idx == 1 and pd.notna(result["train_start"]):
            print(
                f"[Split] Example train {result['train_start'].date()} to {result['train_end'].date()} | "
                f"test {result['test_start'].date()} to {result['test_end'].date()}"
            )
        results.append(result)
        if idx % config.log_every == 0 or idx == len(unique_ids):
            ok = sum(1 for item in results if item["model"] in ("SARIMAX", "SeasonalNaive"))
            print(f"[Progress] {idx}/{len(unique_ids)} series processed; successful={ok}")
    return results


def main() -> None:
    config = parse_args()
    ensure_dirs(config)
    start = time.time()

    print("=" * 80)
    print("Consolidated SARIMAX baseline")
    print(f"Mode: {config.mode}")
    print("=" * 80)

    series_df, supplier_ranking, top_suppliers = prepare_dataset(config)
    df_selected, selected_stats = select_series(series_df, config)
    if df_selected.empty:
        print("[Stop] No eligible series after filtering. Consider lowering --min-nonzero-months.")
        return

    results = fit_all_series(df_selected, selected_stats, config)
    eval_df = summarize_metrics(results)
    print_metric_summary(eval_df, results)
    generate_plots(
        df_selected=df_selected,
        selected_stats=selected_stats,
        results=results,
        eval_df=eval_df,
        config=config,
        supplier_ranking=supplier_ranking,
        top_suppliers=top_suppliers,
    )
    export_results(results, eval_df, config)

    elapsed = time.time() - start
    print("=" * 80)
    print(f"Done in {elapsed / 60:.1f} minutes.")
    print(f"Plots: {config.plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
