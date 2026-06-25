#!/usr/bin/env python3
"""
Standalone TFT forecast validation utility.

This script mirrors the metric logic used in the TFT training pipeline without
requiring model training or checkpoint loading. It compares an actuals CSV with
a forecast CSV, aligns rows by supplier, part, and month, then reports aggregate
MAE, guarded MAPE, WAPE, optional per-horizon metrics, and tolerance buckets.

Expected actuals columns by default:
- SUPP_CD_HASHED
- CAT_ID_NO_HASHED
- MONTHYEAR
- QTY_APPLIED

Expected forecast columns by default:
- SUPP_CD_HASHED
- CAT_ID_NO_HASHED
- MONTHYEAR
- predicted

Column names can be overridden from the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_KEYS = ["SUPP_CD_HASHED", "CAT_ID_NO_HASHED", "MONTHYEAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TFT forecasts against actual shipment quantities.")
    parser.add_argument("--actuals", required=True, help="CSV containing actual observed shipment quantities.")
    parser.add_argument("--forecasts", required=True, help="CSV containing forecasted shipment quantities.")
    parser.add_argument("--output", default="tft_validation_summary.csv", help="Output CSV for validation summary.")
    parser.add_argument("--merged-output", default=None, help="Optional output CSV for aligned row-level validation data.")
    parser.add_argument("--actual-col", default="QTY_APPLIED", help="Actual target column in the actuals CSV.")
    parser.add_argument("--pred-col", default="predicted", help="Prediction column in the forecasts CSV.")
    parser.add_argument("--date-col", default="MONTHYEAR", help="Month/date column used for alignment.")
    parser.add_argument(
        "--id-cols",
        nargs="+",
        default=["SUPP_CD_HASHED", "CAT_ID_NO_HASHED"],
        help="Identifier columns used with the date column to align forecasts and actuals.",
    )
    parser.add_argument(
        "--forecast-horizon-col",
        default=None,
        help="Optional forecast horizon column. If omitted, horizons are inferred per series by month order.",
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
        default=[1, 2, 5, 10, 20, 50],
        help="Absolute-error tolerance buckets to report.",
    )
    return parser.parse_args()


def normalize_key_columns(df: pd.DataFrame, id_cols: list[str], date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    for col in id_cols:
        df[col] = df[col].astype(str).str.strip().fillna("UNKNOWN")
    return df


def load_actuals(path: Path, id_cols: list[str], date_col: str, actual_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = id_cols + [date_col, actual_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Actuals CSV missing columns: {missing}")

    df = normalize_key_columns(df, id_cols, date_col)
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce").fillna(0).clip(lower=0)

    # Actuals may have multiple facilities or rows per supplier-part-month.
    return df.groupby(id_cols + [date_col], as_index=False)[actual_col].sum()


def load_forecasts(path: Path, id_cols: list[str], date_col: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    rename_map = {
        "Supplier": "SUPP_CD_HASHED",
        "Part": "CAT_ID_NO_HASHED",
        "Predicted_Shipments": pred_col,
        "forecast_value": pred_col,
    }
    df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})

    required = id_cols + [date_col, pred_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Forecast CSV missing columns: {missing}")

    df = normalize_key_columns(df, id_cols, date_col)
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
    merged["error"] = merged[pred_col] - merged[actual_col]
    merged["abs_error"] = merged["error"].abs()
    return merged


def infer_horizon(df: pd.DataFrame, id_cols: list[str], date_col: str) -> pd.Series:
    return df.sort_values(id_cols + [date_col]).groupby(id_cols).cumcount() + 1


def compute_metrics(frame: pd.DataFrame, actual_col: str, pred_col: str) -> dict[str, float]:
    actual = frame[actual_col].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    abs_error = np.abs(actual - pred)
    nonzero = actual > 0
    epsilon = 1e-8

    mae = float(np.mean(abs_error)) if len(abs_error) else np.nan
    median_abs_error = float(np.median(abs_error)) if len(abs_error) else np.nan
    mape = float(np.mean(abs_error[nonzero] / (actual[nonzero] + epsilon)) * 100.0) if nonzero.any() else np.nan
    wape = float(np.sum(abs_error) / (np.sum(actual) + epsilon) * 100.0) if len(abs_error) else np.nan
    nonzero_mae = float(np.mean(abs_error[nonzero])) if nonzero.any() else np.nan

    return {
        "n_rows": int(len(frame)),
        "n_nonzero_actuals": int(nonzero.sum()),
        "actual_sum": float(np.sum(actual)),
        "predicted_sum": float(np.sum(pred)),
        "mae": mae,
        "nonzero_mae": nonzero_mae,
        "median_abs_error": median_abs_error,
        "mape_nonzero_actuals": mape,
        "wape": wape,
    }


def tolerance_metrics(frame: pd.DataFrame, tolerances: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if frame.empty:
        for tolerance in tolerances:
            out[f"within_abs_error_{tolerance:g}"] = np.nan
        return out
    for tolerance in tolerances:
        out[f"within_abs_error_{tolerance:g}"] = float((frame["abs_error"] <= tolerance).mean() * 100.0)
    return out


def build_summary(
    merged: pd.DataFrame,
    id_cols: list[str],
    date_col: str,
    actual_col: str,
    pred_col: str,
    horizon_col: str | None,
    tolerances: list[float],
) -> pd.DataFrame:
    rows = []
    overall = {"scope": "overall", "horizon": "all"}
    overall.update(compute_metrics(merged, actual_col, pred_col))
    overall.update(tolerance_metrics(merged, tolerances))
    rows.append(overall)

    if horizon_col:
        if horizon_col not in merged.columns:
            merged[horizon_col] = infer_horizon(merged, id_cols, date_col)
        for horizon, group in merged.groupby(horizon_col, sort=True):
            row = {"scope": "horizon", "horizon": horizon}
            row.update(compute_metrics(group, actual_col, pred_col))
            row.update(tolerance_metrics(group, tolerances))
            rows.append(row)
    else:
        inferred = merged.copy()
        inferred["_inferred_horizon"] = infer_horizon(inferred, id_cols, date_col)
        for horizon, group in inferred.groupby("_inferred_horizon", sort=True):
            row = {"scope": "inferred_horizon", "horizon": int(horizon)}
            row.update(compute_metrics(group, actual_col, pred_col))
            row.update(tolerance_metrics(group, tolerances))
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    date_col = args.date_col
    id_cols = args.id_cols
    actual_col = args.actual_col
    pred_col = args.pred_col

    actuals = load_actuals(Path(args.actuals), id_cols, date_col, actual_col)
    forecasts = load_forecasts(Path(args.forecasts), id_cols, date_col, pred_col)
    merged = align_forecasts(actuals, forecasts, id_cols, date_col, actual_col, pred_col, args.join)

    summary = build_summary(
        merged=merged,
        id_cols=id_cols,
        date_col=date_col,
        actual_col=actual_col,
        pred_col=pred_col,
        horizon_col=args.forecast_horizon_col,
        tolerances=args.tolerances,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    if args.merged_output:
        merged_output = Path(args.merged_output)
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_output, index=False)

    overall = summary.iloc[0]
    print(f"Aligned rows: {int(overall['n_rows']):,}")
    print(f"WAPE: {overall['wape']:.2f}%")
    print(f"MAE: {overall['mae']:.2f}")
    print(f"MAPE, non-zero actuals: {overall['mape_nonzero_actuals']:.2f}%")
    print(f"Summary written to: {output}")


if __name__ == "__main__":
    main()
