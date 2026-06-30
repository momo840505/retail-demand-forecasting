# Error Analysis

This document outlines how forecast errors should be reviewed before using the model for operational decisions.

## Current Evaluation Summary

The final XGBoost model was evaluated across four chronological validation folds.

| Metric | Result |
|---|---:|
| Pooled WAPE | 12.78% |
| Pooled RMSLE | 0.3877 |
| WAPE improvement over best baseline | 24.50% |
| RMSLE improvement over best baseline | 22.56% |

## Error Slices To Review

Recommended slices:

- store;
- product family;
- promotion vs non-promotion days;
- holiday vs non-holiday days;
- high-volume vs low-volume families;
- weekday vs weekend;
- short-term lags vs annual seasonal periods.

## Common Error Patterns

Potential forecasting issues include:

- stockout-like historical sales suppression;
- promotion effects that differ by store;
- holiday effects with local variation;
- sparse sales for low-volume product families;
- abrupt demand shifts not visible in historical lags;
- cold-start limitations for new stores or families.

## Operational Impact

Forecast error matters because replenishment recommendations are derived from expected demand.

Under-forecasting can cause:

- stockouts;
- lost sales;
- poor customer experience.

Over-forecasting can cause:

- excess inventory;
- spoilage for perishable products;
- working-capital inefficiency;
- warehouse and shelf-space pressure.

## Suggested Next Metrics

Add:

- forecast bias by store-family;
- weighted error by revenue or unit volume;
- prediction interval coverage;
- stockout-risk calibration;
- service-level impact simulation;
- inventory cost simulation.

## Recommended Dashboard Upgrade

Add an error-analysis tab showing:

- worst store-family combinations by WAPE;
- fold-by-fold error trend;
- baseline vs XGBoost comparison by product family;
- promotion-day error comparison;
- forecast bias distribution.
