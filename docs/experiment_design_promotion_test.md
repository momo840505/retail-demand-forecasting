# Experiment Design: Does Increasing Promotion Frequency Actually Lift Sales?

## Why this document exists

The historical data in this project already contains a comparison between promoted and non-promoted sales records (`reports/data/promotion_sales_summary.csv`): average sales on promotion are roughly 7x higher than off promotion. It is tempting to read that as "promotions cause a 7x lift." This document explains why that reading is wrong, and designs the randomized experiment that would actually be needed to make a causal claim.

## Part 1 -- What the existing data can and cannot tell us

The promotion flag in the historical dataset was never randomly assigned. Merchandisers choose which products to promote and when, and they do not choose randomly -- promotions are more likely to be placed on higher-traffic categories, around weekends and holidays, and on items that were already expected to sell well. All of that gets bundled into the raw average-sales comparison.

This is confounding, not measurement error, and it cannot be fixed by looking harder at the same data. A few concrete ways it could bias the 7x figure in either direction:

- Promotions concentrated in naturally high-volume categories (e.g. `GROCERY I`, `BEVERAGES`) would inflate the promoted-group average even with zero causal effect.
- Promotions concentrated on weekends (already ~30-40% higher baseline sales, see Part 2) would do the same.
- Conversely, promotions used defensively on slow-moving stock could understate a real effect.

So the existing comparison is kept in this repository as descriptive context (`experiments/promotion_lift_analysis.py`, Part 1) but is explicitly not used as evidence in the design below, and would not be defensible as the basis for a business decision.

## Part 2 -- Designing an actual test

### Hypothesis

Running a promotion on a given store-category combination increases that store-category's daily sales by at least a minimum detectable effect (MDE), relative to a matched control that is not promoted, over the same period.

### Randomization unit

Randomizing at the individual transaction or product level would violate SUTVA (the "stable unit treatment value assumption"): a promoted item competing for the same customer's basket as a non-promoted item in the same store is not independent of the assignment of nearby items. The design instead randomizes at the **store level**, assigning entire stores to treatment (promotion active for the target category) or control (promotion withheld), which avoids within-store interference at the cost of a coarser unit of analysis.

### Primary and guardrail metrics

- **Primary metric:** daily sales (revenue) for the target product category, per store.
- **Guardrail metrics:** total basket size (to catch cannibalization of other categories) and gross margin (to catch a case where a promotion drives volume but destroys margin).

### Estimating the inputs from real data

A sample-size calculation needs a baseline mean and standard deviation. Rather than assume these, `experiments/promotion_lift_analysis.py` computes them directly from `reports/data/daily_sales_summary.csv` (1,684 days of historical network-wide daily sales):

- baseline mean daily sales: **637,556**
- baseline standard deviation: **234,341** (coefficient of variation 0.37 -- daily retail sales are noisy)

The same script breaks this down by day of week and finds a clear seasonality pattern: weekend sales (Saturday/Sunday, ~772K-825K average) run 30-40% above weekday sales (505K-618K average). This is exactly the kind of structure a well-designed experiment should account for rather than treat as noise.

### Why block on day-of-week

A **randomized block design** -- randomizing store assignment within each day-of-week block rather than across the whole pool at once -- removes day-of-week seasonality from the error term instead of leaving it in the noise. Computed from the historical series, blocking on day-of-week reduces the residual variance by **21.1%** (std falls from 234,341 to 208,174). Lower variance means either a smaller required sample for the same MDE, or a smaller detectable effect for the same sample size -- a concrete, data-grounded reason to block rather than randomize naively.

### Sample size / power calculation

Two-sided test, alpha = 0.05, power = 80%, using the standard two-sample size formula `n = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2`:

| Design | MDE | sigma | n per arm (days) |
|---|---|---|---|
| Unblocked (simple randomization) | 5% | 234,341 | 849 |
| Unblocked (simple randomization) | 10% | 234,341 | 213 |
| Blocked by day-of-week | 5% | 208,174 | 670 |
| Blocked by day-of-week | 10% | 208,174 | 168 |

These are network-level illustrative figures (the historical series available in this repository is network-wide, not broken out by store), so a production run of this test would recompute the same calculation on store-category-level historical variance before launch -- store-level daily sales are noisier than the network aggregate, which would raise the required sample size relative to the table above. The methodology and the blocking argument carry over unchanged.

A 10% MDE at 168-213 days per arm is achievable within a single quarter; a 5% MDE at 670-849 days per arm is not a realistic test length for this business. In practice this pushes the recommendation toward either accepting a larger MDE, running the test across more store-category combinations in parallel to increase the effective sample, or accepting a longer commitment before launching it.

### Analysis plan

- Fixed horizon, decided before launch -- no early stopping on interim significance ("peeking" inflates the false-positive rate well above the nominal 5%).
- Primary analysis: two-sample t-test (or a linear model with a day-of-week fixed effect, equivalent to the blocked design above) on the primary metric at the end of the fixed horizon.
- Guardrail metrics reported alongside the primary result even if the primary effect is significant -- a lift that comes from cannibalizing adjacent categories or destroying margin is not a real win.

### Risks and pitfalls to watch for

- **Novelty effects:** a promotion's lift in the first few days can overstate its steady-state effect; the fixed horizon should be long enough to see the effect stabilize, not just its opening spike.
- **Spillover between nearby stores:** stores in the same city/cluster (see `dashboard/data/stores.csv`) may share customers; a control store near several treatment stores could see contaminated results. Cluster-level randomization (promoting entire store clusters together) is a mitigation if this is suspected.
- **Calendar confounds:** holidays and local events affect sales independently of the promotion; the historical series should be checked for known holiday effects before choosing the test window, so the window itself doesn't accidentally overlap a holiday for only one arm.
