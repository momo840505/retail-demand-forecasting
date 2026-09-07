from experiments.promotion_lift_analysis import (
    calculate_weighted_within_day_variance,
)


def test_weighted_within_day_variance_uses_group_counts() -> None:
    day_stats = {
        "A": (10.0, 2.0, 1),
        "B": (20.0, 4.0, 3),
    }
    result = calculate_weighted_within_day_variance(day_stats)
    expected = ((2.0 ** 2) * 1 + (4.0 ** 2) * 3) / 4
    assert result == expected
