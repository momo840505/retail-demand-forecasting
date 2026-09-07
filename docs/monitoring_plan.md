# Monitoring Plan

The `/monitoring-info` endpoint documents the monitoring contract for the
forecasting service. It is not a persistent monitoring backend.

## Data quality checks

- required forecast columns are present;
- forecast dates are parseable;
- store numbers are numeric;
- predicted sales are numeric and non-negative;
- promotion values are numeric;
- store-family coverage matches the deployment manifest.

## Model checks

Once actual sales become available, monitor:

- WAPE by time window;
- RMSLE by time window;
- forecast bias by store and family;
- performance against the simple baseline;
- high-volume family error;
- promotion-day error.

## Service checks

A deployed service should collect:

- request latency;
- HTTP error rate;
- health-check availability;
- forecast data availability;
- forecast row count;
- replenishment requests with incomplete forecast coverage.

These metrics need to be exported to a persistent monitoring system before
the service is considered production-ready. The current repository documents
the required signals but does not claim that a Prometheus, Grafana, or
equivalent monitoring stack is already deployed.

## Retraining triggers

Retraining should be considered when:

- new sales history becomes available;
- forecast bias crosses an agreed threshold;
- WAPE deteriorates relative to the baseline;
- store or product-family coverage changes materially;
- promotion or holiday behaviour changes after a business policy change.

Thresholds should be defined from business tolerance and historical
variation rather than selected after a degradation event.
