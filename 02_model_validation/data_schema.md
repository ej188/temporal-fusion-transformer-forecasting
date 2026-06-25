# Data Schema

The public scripts expect an anonymized monthly supply-chain dataset. The private project data is not included in this repository.

## Required Columns

| Column | Type | Purpose |
| --- | --- | --- |
| `MONTHYEAR` | date/month | Monthly time key. Parsed with `pd.to_datetime`. |
| `ORDERED_QTY` | numeric | Ordered quantity. Used as a known demand/order signal. |
| `QTY_APPLIED` | numeric | Shipment/applied quantity. Used as the TFT target. |
| `SUPP_CD_HASHED` | categorical/string | Anonymized supplier identifier. |
| `CAT_ID_NO_HASHED` | categorical/string | Anonymized part/item identifier. |

Some baseline scripts can also use:

| Column | Type | Purpose |
| --- | --- | --- |
| `FAC_CD_HASHED` | categorical/string | Anonymized facility identifier for facility-level series keys. |

## Cleaning Assumptions

- Invalid dates are dropped.
- Negative order and shipment quantities are clipped to zero.
- The target is transformed with `log1p` for neural forecasting stability.
- Missing numeric feature values created by early lags are filled after leakage-safe shifting.

## Privacy Assumptions

The schema uses hashed identifiers, but hashed row-level records can still be sensitive. Public examples should use synthetic data, aggregate summaries, or schema-only documentation.
