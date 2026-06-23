import pandas as pd
import pytest

from retail_forecasting.features import (
    add_horizon_safe_sales_features,
    build_daily_oil_features,
)


def test_horizon_safe_lag_uses_only_older_sales() -> None:
    test_data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8),
            "store_nbr": [1] * 8,
            "family": ["GROCERY"] * 8,
            "sales": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
        }
    )

    result = add_horizon_safe_sales_features(
        test_data,
        forecast_horizon_days=3,
        lag_days=(3, 4),
        rolling_windows=(2,),
    )

    assert result.loc[3, "sales_lag_3"] == 1.0
    assert result.loc[4, "sales_lag_3"] == 2.0
    assert result.loc[4, "sales_rolling_mean_2_shift_3"] == 1.5


def test_lag_shorter_than_horizon_is_rejected() -> None:
    test_data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8),
            "store_nbr": [1] * 8,
            "family": ["GROCERY"] * 8,
            "sales": [1.0] * 8,
        }
    )

    with pytest.raises(ValueError):
        add_horizon_safe_sales_features(
            test_data,
            forecast_horizon_days=3,
            lag_days=(1, 3),
            rolling_windows=(2,),
        )


def test_oil_features_use_only_horizon_safe_values() -> None:
    oil_data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6),
            "dcoilwtico": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ],
        }
    )

    result = build_daily_oil_features(
        oil_data=oil_data,
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-06"),
        forecast_horizon_days=2,
    )

    assert result["oil_price_lag_2"].tolist() == [
        0.0,
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
    ]

    assert result["oil_price_lag_2_was_missing"].tolist() == [
        1,
        1,
        0,
        0,
        0,
        0,
    ]

    assert result.loc[5, "oil_price_lag_2"] == 40.0
    assert 50.0 not in result["oil_price_lag_2"].tolist()
    assert 60.0 not in result["oil_price_lag_2"].tolist()
