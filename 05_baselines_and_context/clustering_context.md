# Clustering Context

Clustering was used to understand the structure of the supply-chain forecasting problem before and alongside supervised modeling.

## Why It Mattered

The dataset contained many supplier-part combinations with different behavior:

- High-volume stable demand.
- Low-volume intermittent demand.
- Volatile or lumpy shipment patterns.
- Supplier portfolios with different risk profiles.

Segmentation helped the team reason about why a single forecasting method might perform differently across series types.

## Methods Explored

The clustering work included:

- Part-level feature clustering using volume, volatility, and transaction frequency.
- Shape-based clustering of temporal fulfillment patterns.
- Supplier-level aggregation into portfolio-style summaries.
- Outlier/anomaly detection to identify unusual supplier behavior.

## How It Informed Forecasting

The clustering work supported the forecasting effort in three ways:

1. It highlighted that sparse and intermittent demand needed special metric handling.
2. It motivated hierarchical supplier and part features in the TFT pipeline.
3. It provided a business-facing way to describe where forecast risk concentrates.

The final TFT model did not require separate models for each cluster, but the segmentation analysis helped define the operational problem and interpret model performance.
