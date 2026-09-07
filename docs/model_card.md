# Model Card

## Model Overview

This project uses an XGBoost regression model to forecast daily retail demand at the store and product-family level.

The deployed version serves a fixed historical 16-day forecast window and supports replenishment scenarios through a dashboard and API.

## Intended Use

The model is intended for:

- evaluating leakage-aware retail forecasting methods;
- store and product-family demand planning;
- replenishment decision support;
- comparing machine-learning forecasts against transparent baseline methods.

It is not intended for live inventory automation without additional production controls.

## Data

The project uses the Kaggle competition **Store Sales - Time Series Forecasting**, based on Corporación Favorita grocery-store data from Ecuador.

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

## Reproducibility Note

The committed model reports were generated before the project started recording exact package versions inside the model metadata. The current dependency files define the environment for future reruns, and `scripts/train_final_model.py` now records Python, NumPy, pandas, and XGBoost versions whenever the final model is regenerated.

The existing metrics are internally consistent across the checked-in reports, but the exact package versions used for the original committed model artifact cannot be recovered from the repository history alone.

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
