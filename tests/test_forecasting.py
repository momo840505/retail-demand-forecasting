import pandas as pd
import pytest

from retail_forecasting.backtesting import (
    generate_backtest_windows,
)
from retail_forecasting.baselines import (
    add_weekly_seasonal_naive_prediction,
)
from retail_forecasting.metrics import (
    evaluate_forecast,
)


def test_forecast_metrics_are_calculated_correctly() -> None:
    metrics = evaluate_forecast(
        actual_values=[0.0, 2.0, 4.0],
        predicted_values=[0.0, 1.0, 5.0],
    )

    assert metrics["mae"] == pytest.approx(
        2 / 3
    )

    assert metrics["wape_percentage"] == pytest.approx(
        100 * 2 / 6
    )

    assert metrics["rmsle"] >= 0
    assert metrics["rmse"] >= 0


def test_negative_predictions_are_clipped_to_zero() -> None:
    negative_prediction_metrics = evaluate_forecast(
        actual_values=[0.0, 1.0],
        predicted_values=[-5.0, 1.0],
    )

    zero_prediction_metrics = evaluate_forecast(
        actual_values=[0.0, 1.0],
        predicted_values=[0.0, 1.0],
    )

    assert negative_prediction_metrics == zero_prediction_metrics


def test_backtest_windows_are_chronological() -> None:
    windows = generate_backtest_windows(
        final_date="2017-08-15",
        forecast_horizon_days=16,
        fold_count=4,
    )

    assert len(windows) == 4

    assert windows[0].validation_start == pd.Timestamp(
        "2017-06-13"
    )

    assert windows[0].validation_end == pd.Timestamp(
        "2017-06-28"
    )

    assert windows[-1].validation_start == pd.Timestamp(
        "2017-07-31"
    )

    assert windows[-1].validation_end == pd.Timestamp(
        "2017-08-15"
    )


def test_weekly_baseline_uses_only_pre_forecast_sales() -> None:
    source_data = pd.DataFrame(
        {
            "id": range(20),
            "date": pd.date_range(
                "2026-01-01",
                periods=20,
            ),
            "store_nbr": [1] * 20,
            "family": ["GROCERY"] * 20,
            "sales": [
                float(value)
                for value in range(1, 21)
            ],
        }
    )

    result = add_weekly_seasonal_naive_prediction(
        full_data=source_data,
        validation_start="2026-01-13",
        validation_end="2026-01-20",
    )

    predictions = result[
        "weekly_seasonal_naive_prediction"
    ].tolist()

    assert predictions == [
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        6.0,
    ]

    assert (
        result["seasonal_source_date"]
        < pd.Timestamp("2026-01-13")
    ).all()
