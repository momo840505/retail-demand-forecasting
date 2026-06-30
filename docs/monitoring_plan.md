# Monitoring Plan

This portfolio API exposes `/monitoring-info` to document the signals that should be monitored before using the forecasting service in production.

## Data Quality Signals

- required forecast columns are present;
- forecast dates are parseable;
- store numbers are numeric;
- predicted sales are numeric and non-negative;
- promotion flags are numeric;
- store-family coverage matches the deployment manifest.

## Model Quality Signals

- WAPE by validation fold;
- RMSLE by validation fold;
- forecast bias by store-family;
- baseline comparison drift;
- high-volume family error trend;
- promotion-day error trend.

## Operational Signals

- API health status;
- API latency;
- API error rate;
- dashboard data file availability;
- forecast row count;
- replenishment requests with incomplete forecast coverage.

## Retraining Triggers

Retraining should be considered when:

- new sales history becomes available;
- forecast bias exceeds an agreed threshold;
- WAPE deteriorates against baseline;
- store or product-family coverage changes materially;
- promotion or holiday behaviour changes after a business policy change.

## Portfolio Value

The endpoint and this document show that the project treats forecasting as an operational system, not only as a notebook model.
