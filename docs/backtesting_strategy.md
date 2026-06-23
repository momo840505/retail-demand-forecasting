# Backtesting Strategy

## Objective

The project evaluates models using chronological 16-day forecasting
windows that match the final test horizon.

Random train-test splitting is not used because it would allow models to
learn from observations occurring after the validation period.

## Validation Windows

Four consecutive 16-day validation folds are used:

1. 2017-06-13 to 2017-06-28
2. 2017-06-29 to 2017-07-14
3. 2017-07-15 to 2017-07-30
4. 2017-07-31 to 2017-08-15

Each forecast is generated using only information available before or
safely lagged for the applicable target date.

## Baselines

### Zero Forecast

Predicts zero sales for every observation. This provides a minimum
reference for a dataset containing many zero-sales records.

### Lag 16

Uses sales from 16 days earlier. This is directly compatible with the
16-day forecast horizon.

### Weekly Seasonal Naive

Repeats the final seven observed sales days across the complete 16-day
forecast window.

The method does not use actual sales from inside the validation period.

### Shifted 28-Day Mean

Uses a 28-day rolling sales average shifted by the complete 16-day
forecast horizon.

### Lag 364

Uses sales from 364 days earlier to represent approximate annual
seasonality.

## Metrics

- MAE measures average absolute error.
- RMSE gives greater weight to large errors.
- RMSLE evaluates proportional error and reduces domination by very
  large sales values.
- WAPE measures total absolute error relative to total observed sales.

Lower values indicate better forecasting performance.

MAPE is not used because the dataset contains a large proportion of zero
sales observations.
