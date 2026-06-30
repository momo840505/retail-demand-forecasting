# Model Card

## Model Overview

This project uses an XGBoost regression model to forecast daily retail demand at the store and product-family level.

The deployed portfolio version produces a historical 16-day forecast window and supports replenishment decision scenarios through a dashboard and API.

## Intended Use

The model is intended for:

- portfolio demonstration of leakage-aware forecasting;
- store and product-family demand planning;
- replenishment decision support;
- comparing machine-learning forecasts against transparent baseline methods.

It is not intended for live inventory automation without additional production controls.

## Data

The project uses the Kaggle Corporacion Favorita Grocery Sales Forecasting dataset.

Signals include:

- historical unit sales;
- store identifiers;
- product families;
- promotion flags;
- calendar variables;
- holidays and events;
- oil-price history;
- store metadata.

The original raw competition files are not committed to the repository.

## Feature Engineering

Feature engineering is forecast-horizon safe. The forecast horizon is 16 days, so lag and rolling features use only information available before the forecast period begins.

Examples:

- `sales_lag_16`
- `sales_lag_21`
- `sales_lag_28`
- `sales_lag_35`
- `sales_lag_364`
- shifted rolling means;
- shifted rolling standard deviations;
- known calendar and promotion features.

## Evaluation

The model is evaluated with chronological validation rather than random splitting.

| Metric | Result |
|---|---:|
| Pooled WAPE | 12.78% |
| Pooled RMSLE | 0.3877 |
| WAPE improvement over best baseline | 24.50% |
| RMSLE improvement over best baseline | 22.56% |
| Backtesting folds | 4 |
| Forecast horizon | 16 days |

## Baselines

The model is compared against:

- zero forecast;
- lag-16 forecast;
- lag-364 forecast;
- weekly seasonal naive forecast;
- shifted 28-day mean forecast.

## Known Limitations

- The deployed demo uses a fixed historical forecast window.
- Final competition test labels are unavailable.
- Inventory levels and supplier lead times are user-defined assumptions.
- Forecast intervals and calibrated uncertainty are not included yet.
- Model monitoring and scheduled retraining are planned but not deployed.

## Recommended Production Extensions

- prediction intervals;
- probabilistic stockout risk;
- scheduled retraining;
- data drift monitoring;
- forecast bias monitoring by store and product family;
- persistent inventory and order history;
- integration with real supplier lead-time data.
