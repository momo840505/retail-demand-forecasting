"""Promotion lift: what the historical data can and cannot tell us, plus a
sample-size / power calculation for a proper randomized test.

This script has two parts:

1. Observational comparison (reports/data/promotion_sales_summary.csv)
   Reports the raw difference in average sales between promoted and
   non-promoted records -- and explains why this number is NOT evidence
   that promotions *cause* higher sales. The data was never randomized:
   promotions are placed on products/days a merchandiser already expected
   to sell well (higher-traffic categories, weekends, holidays), so the
   comparison is confounded. This is a classic case where a large,
   easy-to-compute effect size is misleading without a designed experiment.

2. Experiment design (the actual deliverable)
   Uses the *historical daily sales series* (reports/data/daily_sales_summary.csv)
   only to estimate real-world variability -- the baseline mean and standard
   deviation of daily sales, and the day-of-week seasonality -- as legitimate,
   data-grounded inputs to a standard sample-size / power calculation for a
   future randomized promotion test. It also demonstrates, numerically, why
   blocking the design on day-of-week (a randomized block design) meaningfully
   increases statistical power compared to a naive, unblocked randomization.

Run with:  python experiments/promotion_lift_analysis.py
No external dependencies beyond the Python standard library.
"""

import csv
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "reports" / "data"

# Two-sided test, alpha = 0.05, power = 80% -- standard defaults.
Z_ALPHA = 1.959964
Z_BETA = 0.841621

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def load_promotion_summary() -> list[dict]:
    with open(DATA_DIR / "promotion_sales_summary.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_daily_sales() -> list[dict]:
    with open(DATA_DIR / "daily_sales_summary.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_observational_comparison() -> None:
    print("=" * 78)
    print("PART 1 -- Observational promotion comparison (NOT a randomized test)")
    print("=" * 78)
    rows = load_promotion_summary()
    for row in rows:
        print(
            f"  {row['promotion_status']:<20s} "
            f"avg_sales={float(row['average_sales']):>10,.2f}  "
            f"records={int(row['record_count']):>10,d}"
        )
    print(
        "\n  Raw ratio (on-promotion avg / not-on-promotion avg) looks like a huge "
        "lift, but promoted items/days were CHOSEN, not randomly assigned.\n"
        "  Category, day-of-week, seasonality and store effects are all bundled "
        "into that number. Treat it as descriptive context only -- it is not\n"
        "  usable as evidence of a causal promotion effect, and it is not the "
        "basis for the sample-size calculation below."
    )
    print()


def compute_daily_stats() -> tuple[float, float, dict[str, tuple[float, float, int]]]:
    rows = load_daily_sales()
    sales = [float(r["total_sales"]) for r in rows]
    overall_mean = statistics.mean(sales)
    overall_std = statistics.pstdev(sales)

    by_dow: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        d = date.fromisoformat(r["date"])
        by_dow[DAYS_OF_WEEK[d.weekday()]].append(float(r["total_sales"]))

    dow_stats = {
        day: (statistics.mean(vals), statistics.pstdev(vals), len(vals))
        for day, vals in by_dow.items()
    }
    return overall_mean, overall_std, dow_stats


def sample_size_per_group(sigma: float, delta: float) -> int:
    """Two-sample z-test sample size per group, two-sided alpha=0.05, power=80%."""
    n = 2 * ((Z_ALPHA + Z_BETA) ** 2) * (sigma ** 2) / (delta ** 2)
    return math.ceil(n)


def print_experiment_design() -> None:
    print("=" * 78)
    print("PART 2 -- Sample size for a real randomized promotion test")
    print("=" * 78)
    overall_mean, overall_std, dow_stats = compute_daily_stats()

    print(f"  Baseline daily sales -- mean: {overall_mean:,.0f}, std: {overall_std:,.0f}")
    print(f"  Coefficient of variation: {overall_std / overall_mean:.3f}\n")

    print("  Day-of-week breakdown (evidence of seasonality to block on):")
    for day in DAYS_OF_WEEK:
        m, s, n = dow_stats[day]
        print(f"    {day:<10s} n={n:>4d}  mean={m:>11,.0f}  std={s:>10,.0f}")

    within_dow_var = statistics.mean([s ** 2 for _, s, _ in dow_stats.values()])
    overall_var = overall_std ** 2
    variance_reduction = 1 - within_dow_var / overall_var
    blocked_std = math.sqrt(within_dow_var)

    print(
        f"\n  Blocking on day-of-week reduces variance by {variance_reduction:.1%} "
        f"({overall_std:,.0f} -> {blocked_std:,.0f} std),\n"
        "  because a large share of daily variability is explained by which "
        "day of the week it is (weekends run 30-40% above weekdays)."
    )

    print("\n  Sample size per arm (days), two-sided alpha=0.05, power=80%:\n")
    header = f"  {'Design':<32s} {'MDE':>6s} {'sigma':>10s} {'n/arm (days)':>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, sigma in [
        ("Unblocked (simple randomization)", overall_std),
        ("Blocked by day-of-week", blocked_std),
    ]:
        for mde_pct in (0.05, 0.10):
            delta = overall_mean * mde_pct
            n = sample_size_per_group(sigma, delta)
            print(f"  {label:<32s} {mde_pct:>5.0%} {sigma:>10,.0f} {n:>14,d}")
    print()


if __name__ == "__main__":
    print_observational_comparison()
    print_experiment_design()
