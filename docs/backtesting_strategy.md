# Backtesting Strategy

## Objective

The project evaluates forecasting models with chronological 16-day
validation windows matching the final test horizon.

Random train-test splitting is not used because it would allow later
observations to influence evaluation of earlier forecasts.

## Outer Validation Windows

Four consecutive 16-day outer validation folds are used:

1. 2017-06-13 to 2017-06-28
2. 2017-06-29 to 2017-07-14
3. 2017-07-15 to 2017-07-30
4. 2017-07-31 to 2017-08-15

Outer validation data is used only for final performance measurement.

## Nested Model Selection

Within each outer training period, the final 16 days form an inner
validation set.

The process for every fold is:

1. Train a candidate model on the inner training period.
2. Use inner validation for early stopping.
3. Record the selected number of boosting estimators.
4. Retrain a fresh model on the complete outer training period.
5. Evaluate once on the untouched outer validation period.

This prevents the outer validation set from influencing model complexity
or early-stopping decisions.

## Training Window

Each outer fold uses the most recent 730 days of available training data.

This balances:

- recent demand patterns;
- sufficient seasonal history;
- computational efficiency;
- reduced influence from outdated retail behaviour.

## Baselines

The evaluated baselines are:

- zero forecast;
- sales lag 16;
- weekly seasonal naive;
- 28-day rolling mean shifted by 16 days;
- sales lag 364.

## XGBoost Model

The XGBoost model:

- predicts `log1p(sales)`;
- supports native categorical features;
- uses histogram-based tree construction;
- selects estimator count through inner early stopping;
- converts predictions using `expm1`;
- clips final forecasts to zero or above.

## Metrics

- MAE measures average absolute unit error.
- RMSE gives greater weight to large errors.
- RMSLE evaluates proportional forecasting error.
- WAPE measures total absolute error relative to total observed sales.

Lower values indicate better performance.

MAPE is not used because many observations contain zero sales.

## Interpretation

Feature importance represents how the fitted model uses variables for
prediction. It does not demonstrate that a feature causes changes in
sales.
