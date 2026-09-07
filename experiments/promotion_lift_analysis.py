"""Promotion diagnostics and inputs for a future randomized experiment.

The historical promotion flag is observational. The raw promoted versus
non-promoted difference is useful as descriptive context, but it is not a
causal estimate because promotion assignment was not randomized.

This script reports that descriptive comparison and quantifies day-of-week
variation in network sales. It intentionally does not report a required
sample size for a store-randomized experiment because the committed
network-level series does not contain the store-level repeated-measures
variance needed for that calculation.
"""

import csv
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "reports" / "data"

DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def load_promotion_summary() -> list[dict[str, str]]:
    """Load the descriptive promotion summary."""
    with (DATA_DIR / "promotion_sales_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        return list(csv.DictReader(input_file))


def load_daily_sales() -> list[dict[str, str]]:
    """Load the network-level daily sales summary."""
    with (DATA_DIR / "daily_sales_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        return list(csv.DictReader(input_file))


def print_observational_comparison() -> None:
    """Print descriptive promoted and non-promoted sales averages."""
    rows = load_promotion_summary()
    averages: dict[str, float] = {}

    print("=" * 72)
    print("PROMOTION STATUS - DESCRIPTIVE COMPARISON")
    print("=" * 72)

    for row in rows:
        label = row["promotion_status"]
        average_sales = float(row["average_sales"])
        averages[label] = average_sales

        print(
            f"{label:<20s} "
            f"average_sales={average_sales:>10,.2f} "
            f"records={int(row['record_count']):>10,d}"
        )

    promoted = averages.get("On promotion")
    not_promoted = averages.get("Not on promotion")

    if promoted is not None and not_promoted:
        print(
            f"Raw average ratio: "
            f"{promoted / not_promoted:.2f}x"
        )

    print(
        "\nThis comparison is observational. Promotion assignment was "
        "not randomized, so category mix, store mix, calendar effects, "
        "and merchandising decisions can all contribute to the gap."
    )


def compute_day_of_week_stats() -> tuple[
    float,
    float,
    dict[str, tuple[float, float, int]],
]:
    """Calculate network sales statistics by day of week."""
    rows = load_daily_sales()
    sales = [float(row["total_sales"]) for row in rows]

    overall_mean = statistics.mean(sales)
    overall_std = statistics.pstdev(sales)

    by_day: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        observed_date = date.fromisoformat(row["date"])
        day_name = DAYS_OF_WEEK[observed_date.weekday()]
        by_day[day_name].append(float(row["total_sales"]))

    day_stats = {
        day_name: (
            statistics.mean(values),
            statistics.pstdev(values),
            len(values),
        )
        for day_name, values in by_day.items()
    }

    return overall_mean, overall_std, day_stats


def print_design_inputs() -> None:
    """Print historical variation relevant to experiment planning."""
    overall_mean, overall_std, day_stats = (
        compute_day_of_week_stats()
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT DESIGN INPUTS")
    print("=" * 72)
    print(
        f"Network daily sales mean={overall_mean:,.0f}, "
        f"std={overall_std:,.0f}"
    )

    for day_name in DAYS_OF_WEEK:
        mean_value, std_value, count = day_stats[day_name]
        print(
            f"{day_name:<10s} "
            f"n={count:>4d} "
            f"mean={mean_value:>11,.0f} "
            f"std={std_value:>10,.0f}"
        )

    within_day_variance = statistics.mean(
        standard_deviation ** 2
        for _, standard_deviation, _ in day_stats.values()
    )

    variance_reduction = (
        1.0
        - within_day_variance / (overall_std ** 2)
    )

    print(
        "\nRemoving the day-of-week mean explains "
        f"{variance_reduction:.1%} of network-level variance."
    )

    print(
        "\nA store-randomized promotion test needs a power calculation "
        "based on store-category repeated measures. The network-wide "
        "daily series used here is not the correct sampling unit for "
        "that calculation."
    )


if __name__ == "__main__":
    print_observational_comparison()
    print_design_inputs()
