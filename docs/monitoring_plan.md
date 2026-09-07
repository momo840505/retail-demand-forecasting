# Monitoring Plan

The `/monitoring-info` endpoint documents the checks and signals that would be
needed around the forecasting service. It is not a persistent telemetry or
alerting backend.

## Current Scope

The API currently provides:

- a `/health` endpoint;
- validation of prepared forecast data when it is loaded;
- a documented monitoring contract through `/monitoring-info`.

The endpoint returns an explicit implementation status so callers can
distinguish documented signals from telemetry that is actually being
collected.

## Data Quality Checks

- required forecast columns are present;
- forecast dates are parseable;
- store numbers are numeric;
- predicted sales are numeric and non-negative;
- promotion values are numeric;
- store-family coverage matches the deployment manifest.

## Model Signals To Add

Once actual sales become available, monitor:

- WAPE by time window;
- RMSLE by time window;
- forecast bias by store and family;
- performance against the simple baseline;
- high-volume family error;
- promotion-day error.

## Service Signals To Add

A production deployment should collect:

- request latency;
- HTTP error rate;
- health-check availability;
- forecast-data availability;
- forecast row count;
- replenishment requests with incomplete forecast coverage.

These signals need to be exported to a persistent monitoring system before
the service should be treated as production-monitored.

## Retraining Triggers

Retraining should be considered when:

- new sales history becomes available;
- forecast bias crosses an agreed threshold;
- WAPE deteriorates relative to the baseline;
- store or product-family coverage changes materially;
- promotion or holiday behaviour changes after a business policy change.

Thresholds should be defined from business tolerance and historical variation
rather than selected after a degradation event.
