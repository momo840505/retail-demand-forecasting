# Feature Availability and Leakage Policy

## Forecast Horizon

The project produces a direct 16-day retail demand forecast.

Every model feature must be available when the forecast is created.

## Known-in-Advance Features

The following information is available before the forecast period:

- forecast date;
- calendar variables;
- store metadata;
- planned promotion counts;
- known national, regional, and local calendar events.

## Historical Sales Features

Sales features use only sufficiently old observations:

- sales lag 16;
- sales lag 21;
- sales lag 28;
- sales lag 35;
- sales lag 364.

Rolling statistics are shifted by the complete 16-day forecast horizon.

This prevents a prediction for any day within the forecast window from
using sales that would not yet be observed when the forecast is created.

## Oil-Price Features

Realised future oil prices are not used.

Oil-price variables are shifted by 16 days:

- oil price lag 16;
- seven-day oil-price mean shifted by 16 days;
- lagged oil-price missing indicator.

Missing values are filled using earlier available observations only.

## Holiday Mapping

Calendar events are applied according to geographic scope:

- national events apply to all stores;
- regional events apply to stores in the matching state;
- local events apply to stores in the matching city.

Original holiday dates marked as transferred are excluded. Their
corresponding transfer dates are represented separately.

## Excluded Transactions Feature

Historical transaction counts are useful for exploratory analysis but are
not included in the forecasting model.

Future transaction counts are unavailable during the competition test
period. Including transactions during training would create a
training-serving mismatch and unrealistically optimistic validation.

## Leakage Rule

No feature may include sales, oil-price, transaction, or other realised
information from within the 16-day forecast window.
