# Error Analysis

## Current backtest result

The XGBoost model was evaluated on four untouched 16-day periods.

| Fold | WAPE | RMSLE |
|---|---:|---:|
| 1 | 12.54% | 0.3851 |
| 2 | 10.68% | 0.3749 |
| 3 | 12.32% | 0.3907 |
| 4 | 15.67% | 0.3999 |

Pooled WAPE is 12.78% and pooled RMSLE is 0.3877.

The fourth fold is the weakest period on WAPE. That variation matters
because a pooled metric can hide periods where replenishment decisions would
be less reliable.

## What the committed results support

The model beats the strongest simple baseline on pooled WAPE and RMSLE, but
the error is not constant over time.

The repository does not commit row-level XGBoost outer-fold predictions, so
it cannot currently reproduce store- or family-level error slices from the
checked-in artifacts alone. This document does not claim those diagnostics
have already been run.

## Diagnostics to add on the next backtest

The next backtest should persist row-level outer-fold predictions and report:

- error by store;
- error by product family;
- promotion versus non-promotion days;
- holiday versus regular days;
- volume decile;
- forecast bias;
- zero-demand frequency;
- worst store-family combinations.

Forecast bias should be reported alongside WAPE:

`sum(prediction - actual) / sum(actual)`

Systematic under-forecasting and over-forecasting have different inventory
costs even when their absolute errors are similar.

## Operational implication

Under-forecasting can create stockouts and lost sales. Over-forecasting can
increase holding cost, spoilage, and working capital.

The replenishment layer currently consumes point forecasts. A production
version should add prediction intervals or quantile forecasts before using
the model to set service-level inventory targets.
