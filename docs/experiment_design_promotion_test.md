# Promotion Experiment Design

## Question

Would a promotion for a target product category increase sales without
reducing gross margin or simply shifting demand away from nearby categories?

## What the historical data can show

The historical promotion flag is observational. Promotions were chosen by
the retailer rather than randomly assigned, so the difference between
promoted and non-promoted records is not a causal estimate.

`experiments/promotion_lift_analysis.py` keeps the raw comparison as
descriptive context and also measures day-of-week variation in the
network-level sales series. The network series is useful for demonstrating
calendar variation, but it is not used to claim a sample size for a
store-randomized experiment.

## Proposed randomization

Treatment would be assigned at the store level for a selected product
category:

- treatment stores run the planned promotion;
- control stores keep the normal merchandising plan;
- assignment is stratified by comparable store characteristics before
  randomization.

Store-level assignment reduces within-store interference. Geographic
spillover between nearby stores still needs to be checked before launch.

## Metrics

Primary metric:

- daily category sales per store.

Guardrails:

- gross margin;
- total basket value;
- adjacent-category sales;
- stock availability.

A promotion should not be considered successful if the apparent sales lift
is explained by margin loss, cannibalisation, or stock-management
differences.

## Power calculation

A defensible power calculation must use the same unit structure as the
planned experiment. For a store-randomized design this requires
store-category repeated measures, including:

- between-store variance;
- within-store day-to-day variance;
- serial correlation;
- treatment duration;
- number of stores per arm;
- any stratification or baseline adjustment.

The committed summary data is network-wide, so it does not contain enough
information to estimate those quantities correctly. This repository
therefore does not report a numeric required sample size for the
store-randomized design.

Before launch, power should be computed from historical store-category data
using a cluster-aware formula or simulation, and the analysis plan should be
fixed before experiment outcomes are examined.

## Analysis plan

The main analysis would estimate the treatment effect using store-day data
with pre-specified calendar controls and store effects. Standard errors should
be clustered by store to account for repeated observations from the same
randomization unit. If a pre-period is available, baseline adjustment can
reduce variance.

Confidence intervals and guardrail metrics should be reported with the
primary estimate. Repeated significance checking should not be used unless a
sequential-testing design is specified in advance.

## Main risks

- spillover between nearby treatment and control stores;
- stockouts that cap observed sales;
- holiday or payday effects;
- inconsistent promotion execution;
- novelty effects early in the test;
- category cannibalisation.
