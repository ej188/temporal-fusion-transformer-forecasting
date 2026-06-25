#!/usr/bin/env python3
"""
Public-safe forecast performance evaluation utility.

This script compares actual shipment quantities with forecasted quantities and
creates project-summary artifacts:
- overall performance metrics
- monthly performance metrics
- optional per-horizon metrics
- tolerance-bucket accuracy chart
- actual-vs-predicted scatter chart
- monthly actual-vs-predicted volume chart

It intentionally uses CLI arguments instead of hard-coded private paths.

Default actuals columns:
- SUPP_CD_HASHED
- CAT_ID_NO_HASHED
- MONTHYEAR
- QTY_APPLIED

Default forecast columns:
- SUPP_CD_HASHED
- CAT_ID_NO_HASHED
- MONTHYEAR
- predicted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ID_COLS = ["SUPP_CD_HASHED", "CAT_ID_NO_HASHED"]
DEFAULT_TOLERANCES = [1, 2, 5, 10, 20, 50]


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate forecast performance and generate public summary artifacts."
    )
    parser.add_argument("--actuals", required=True, help="CSV containing observed shipment quantities.")
    parser.add_argument("--forecasts", required=True, help="CSV containing forecasted shipment quantities.")
    parser.add_argument(
        "--output-dir",
        default="forecast_performance_outputs",
        help="Directory for summary CSVs, aligned rows, and charts.",
    )
    parser.add_argument("--actual-col", default="QTY_APPLIED", help="Actual target column in the actuals CSV.")
    parser.add_argument("--pred-col", default="predicted", help="Prediction column in the forecasts CSV.")
    parser.add_argument("--date-col", default="MONTHYEAR", help="Month/date column used for alignment.")
    parser.add_argument(
        "--id-cols",
        nargs="+",
        default=DEFAULT_ID_COLS,
        help="Identifier columns used with the date column to align forecasts and actuals.",
    )
    parser.add_argument(
        "--forecast-horizon-col",
        default=None,
        help="Optional forecast horizon column. If missing, horizons are inferred by series month order.",
    )
    parser.add_argument(
        "--join",
        choices=["left", "inner"],
        default="left",
        help="Use left join to keep all forecasts and treat missing actuals as zero, or inner join for strict overlap.",
    )
    parser.add_argument(
        "--tolerances",
        nargs="*",
        type=float,
        default=DEFAULT_TOLERANCES,
        help="Absolute-error tolerance buckets to report and plot.",
    )
    parser.add_argument(
        "--top-error-rows",
        type=int,
        default=100,
        help="Number of largest absolute-error rows to write for inspection.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Maximum points shown in the actual-vs-predicted scatter plot.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_keys(df: pd.DataFrame, id_cols: list[str], date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    for col in id_cols:
        df[col] = df[col].astype(str).str.strip().replace({"": "UNKNOWN"})
    return df


def load_actuals(path: Path, id_cols: list[str], date_col: str, actual_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = id_cols + [date_col, actual_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Actuals CSV missing columns: {missing}")

    df = normalize_keys(df, id_cols, date_col)
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce").fillna(0).clip(lower=0)
    return df.groupby(id_cols + [date_col], as_index=False)[actual_col].sum()


def load_forecasts(path: Path, id_cols: list[str], date_col: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    rename_map = {
        "Supplier": "SUPP_CD_HASHED",
        "Part": "CAT_ID_NO_HASHED",
        "Predicted_Shipments": pred_col,
        "forecast_value": pred_col,
        "prediction": pred_col,
    }
    df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})

    required = id_cols + [date_col, pred_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Forecast CSV missing columns: {missing}")

    df = normalize_keys(df, id_cols, date_col)
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.dropna(subset=[pred_col])
    df[pred_col] = df[pred_col].clip(lower=0)
    return df


def align_forecasts(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    id_cols: list[str],
    date_col: str,
    actual_col: str,
    pred_col: str,
    join: str,
) -> pd.DataFrame:
    merged = forecasts.merge(actuals, on=id_cols + [date_col], how=join)
    if join == "left":
        merged[actual_col] = merged[actual_col].fillna(0)

    merged = merged.dropna(subset=[actual_col, pred_col]).copy()
    merged = merged.rename(columns={actual_col: "actual", pred_col: "predicted"})
    merged["error"] = merged["predicted"] - merged["actual"]
    merged["abs_error"] = merged["error"].abs()
    merged["ape"] = np.where(merged["actual"] > 0, merged["abs_error"] / merged["actual"] * 100.0, np.nan)
    return merged


def infer_horizon(df: pd.DataFrame, id_cols: list[str], date_col: str) -> pd.Series:
    return df.sort_values(id_cols + [date_col]).groupby(id_cols).cumcount() + 1


def safe_r2(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if len(actual_values) < 2 or np.std(actual_values) == 0 or np.std(predicted_values) == 0:
        return np.nan
    return float(np.corrcoef(actual_values, predicted_values)[0, 1] ** 2)


def metric_row(
    frame: pd.DataFrame,
    id_cols: list[str],
    scope: str,
    label: str,
    tolerances: list[float],
) -> dict[str, float | str]:
    abs_error = frame["abs_error"]
    actual = frame["actual"]
    predicted = frame["predicted"]
    nonzero = actual > 0
    epsilon = 1e-8

    row: dict[str, float | str] = {
        "scope": scope,
        "label": label,
        "n_rows": int(len(frame)),
        "n_series": int(frame[id_cols].astype(str).agg("/".join, axis=1).nunique())
        if all(col in frame.columns for col in id_cols)
        else np.nan,
        "actual_sum": float(actual.sum()),
        "predicted_sum": float(predicted.sum()),
        "bias": float((predicted - actual).mean()) if len(frame) else np.nan,
        "mae": float(abs_error.mean()) if len(frame) else np.nan,
        "median_abs_error": float(abs_error.median()) if len(frame) else np.nan,
        "nonzero_mae": float(abs_error[nonzero].mean()) if nonzero.any() else np.nan,
        "mape_nonzero_actuals": float(frame.loc[nonzero, "ape"].mean()) if nonzero.any() else np.nan,
        "wape": float(abs_error.sum() / (actual.sum() + epsilon) * 100.0) if len(frame) else np.nan,
        "r2_corr": safe_r2(actual, predicted),
    }

    for tolerance in tolerances:
        row[f"within_abs_error_{tolerance:g}"] = (
            float((abs_error <= tolerance).mean() * 100.0) if len(frame) else np.nan
        )
    return row


def build_summary(
    merged: pd.DataFrame,
    id_cols: list[str],
    date_col: str,
    horizon_col: str | None,
    tolerances: list[float],
) -> pd.DataFrame:
    rows = [metric_row(merged, id_cols, "overall", "all", tolerances)]

    for month, group in merged.groupby(date_col, sort=True):
        rows.append(metric_row(group, id_cols, "month", month.strftime("%Y-%m"), tolerances))

    if horizon_col:
        if horizon_col not in merged.columns:
            merged = merged.copy()
            merged[horizon_col] = infer_horizon(merged, id_cols, date_col)
        for horizon, group in merged.groupby(horizon_col, sort=True):
            rows.append(metric_row(group, id_cols, "horizon", str(horizon), tolerances))

    return pd.DataFrame(rows)


def write_outputs(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    top_error_rows: int,
    date_col: str,
) -> None:
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    merged.to_csv(output_dir / "aligned_forecast_actuals.csv", index=False)

    top_errors = merged.sort_values("abs_error", ascending=False).head(top_error_rows)
    top_errors.to_csv(output_dir / "top_absolute_errors.csv", index=False)

    monthly = (
        merged.groupby(date_col, as_index=False)
        .agg(actual=("actual", "sum"), predicted=("predicted", "sum"), abs_error=("abs_error", "sum"))
    )
    monthly["wape"] = monthly["abs_error"] / (monthly["actual"] + 1e-8) * 100.0
    monthly.to_csv(output_dir / "monthly_volume_performance.csv", index=False)


def plot_tolerance_buckets(summary: pd.DataFrame, tolerances: list[float], output_dir: Path) -> None:
    plt = get_pyplot()
    overall = summary.query("scope == 'overall'").iloc[0]
    labels = [f"+/-{t:g}" for t in tolerances]
    values = [overall[f"within_abs_error_{t:g}"] for t in tolerances]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#2D7DD2", edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, max(values + [1]) * 1.18)
    ax.set_ylabel("% of predictions within tolerance")
    ax.set_xlabel("Absolute error tolerance")
    ax.set_title("Forecast Accuracy by Error Tolerance")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_tolerance_buckets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(merged: pd.DataFrame, output_dir: Path, sample_size: int) -> None:
    plt = get_pyplot()
    sample = merged
    if sample_size > 0 and len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)

    max_axis = max(sample["actual"].max(), sample["predicted"].max(), 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample["actual"], sample["predicted"], s=10, alpha=0.25, color="#4A4F57")
    ax.plot([0, max_axis], [0, max_axis], color="#E74C3C", linewidth=1.5, linestyle="--")
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_xlabel("Actual quantity")
    ax.set_ylabel("Predicted quantity")
    ax.set_title("Actual vs. Predicted Shipments")
    ax.grid(alpha=0.2, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_volume(merged: pd.DataFrame, date_col: str, output_dir: Path) -> None:
    plt = get_pyplot()
    monthly = (
        merged.groupby(date_col, as_index=False)
        .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
        .sort_values(date_col)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(monthly[date_col], monthly["actual"], label="Actual", linewidth=2.2, color="#1A1A1A")
    ax.plot(monthly[date_col], monthly["predicted"], label="Predicted", linewidth=2.2, color="#2D7DD2")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total quantity")
    ax.set_title("Monthly Actual vs. Predicted Volume")
    ax.legend()
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    actuals = load_actuals(Path(args.actuals), args.id_cols, args.date_col, args.actual_col)
    forecasts = load_forecasts(Path(args.forecasts), args.id_cols, args.date_col, args.pred_col)
    merged = align_forecasts(
        actuals=actuals,
        forecasts=forecasts,
        id_cols=args.id_cols,
        date_col=args.date_col,
        actual_col=args.actual_col,
        pred_col=args.pred_col,
        join=args.join,
    )

    summary = build_summary(
        merged=merged,
        id_cols=args.id_cols,
        date_col=args.date_col,
        horizon_col=args.forecast_horizon_col,
        tolerances=args.tolerances,
    )

    write_outputs(merged, summary, output_dir, args.top_error_rows, args.date_col)
    plot_tolerance_buckets(summary, args.tolerances, output_dir)
    plot_actual_vs_predicted(merged, output_dir, args.sample_size)
    plot_monthly_volume(merged, args.date_col, output_dir)

    overall = summary.query("scope == 'overall'").iloc[0]
    print(f"Aligned rows: {int(overall['n_rows']):,}")
    print(f"Series evaluated: {int(overall['n_series']):,}" if pd.notna(overall["n_series"]) else "Series evaluated: n/a")
    print(f"WAPE: {overall['wape']:.2f}%")
    print(f"MAE: {overall['mae']:.2f}")
    print(f"Median absolute error: {overall['median_abs_error']:.2f}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
