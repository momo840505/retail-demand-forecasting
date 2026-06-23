from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


FORECAST_HORIZON_DAYS = 16

DEFAULT_LAG_DAYS = (
    16,
    21,
    28,
    35,
    364,
)

DEFAULT_ROLLING_WINDOWS = (
    7,
    28,
)

HOLIDAY_FEATURE_COLUMNS = [
    "is_special_day",
    "is_holiday",
    "is_event",
    "is_work_day",
    "is_transfer",
    "is_bridge",
    "is_additional",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
]


def add_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add calendar variables known before the forecast period."""
    result = dataframe.copy()
    result["date"] = pd.to_datetime(result["date"])

    result["year"] = result["date"].dt.year.astype("int16")
    result["quarter"] = result["date"].dt.quarter.astype("int8")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day"] = result["date"].dt.day.astype("int8")
    result["day_of_week"] = result["date"].dt.dayofweek.astype("int8")
    result["day_of_year"] = result["date"].dt.dayofyear.astype("int16")
    result["week_of_year"] = (
        result["date"].dt.isocalendar().week.astype("int16")
    )

    result["is_weekend"] = (
        result["day_of_week"].isin([5, 6]).astype("int8")
    )

    result["is_month_start"] = (
        result["date"].dt.is_month_start.astype("int8")
    )

    result["is_month_end"] = (
        result["date"].dt.is_month_end.astype("int8")
    )

    result["is_year_start"] = (
        result["date"].dt.is_year_start.astype("int8")
    )

    result["is_year_end"] = (
        result["date"].dt.is_year_end.astype("int8")
    )

    minimum_date = result["date"].min()

    result["time_index"] = (
        result["date"] - minimum_date
    ).dt.days.astype("int32")

    return result


def build_daily_oil_features(
    oil_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    forecast_horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Create horizon-safe daily oil-price features.

    For a direct 16-day forecast, each row may only use oil information
    observed at least 16 days earlier. This prevents future realised oil
    prices from leaking into training or validation features.
    """
    result = oil_data.copy()
    result["date"] = pd.to_datetime(result["date"])

    result = (
        result.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )

    daily_calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                start=start_date,
                end=end_date,
                freq="D",
            )
        }
    )

    result = daily_calendar.merge(
        result[["date", "dcoilwtico"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    shifted_oil_price = result["dcoilwtico"].shift(
        forecast_horizon_days
    )

    missing_indicator_column = (
        f"oil_price_lag_{forecast_horizon_days}_was_missing"
    )

    lagged_price_column = (
        f"oil_price_lag_{forecast_horizon_days}"
    )

    rolling_mean_column = (
        f"oil_price_7_day_mean_shift_{forecast_horizon_days}"
    )

    result[missing_indicator_column] = (
        shifted_oil_price.isna().astype("int8")
    )

    horizon_safe_oil_price = (
        shifted_oil_price
        .ffill()
        .fillna(0.0)
        .astype("float32")
    )

    result[lagged_price_column] = horizon_safe_oil_price

    result[rolling_mean_column] = (
        horizon_safe_oil_price
        .rolling(window=7, min_periods=1)
        .mean()
        .astype("float32")
    )

    return result[
        [
            "date",
            lagged_price_column,
            rolling_mean_column,
            missing_indicator_column,
        ]
    ]

def build_store_holiday_features(
    stores_data: pd.DataFrame,
    holidays_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map national, regional, and local events to applicable stores.

    Holiday records marked as transferred are excluded from their
    original date because the holiday did not occur on that date.
    """
    stores_data = stores_data.copy()
    holidays_data = holidays_data.copy()

    holidays_data["date"] = pd.to_datetime(holidays_data["date"])

    active_events = holidays_data.loc[
        ~holidays_data["transferred"].astype(bool)
    ].copy()

    holiday_records: list[dict[str, object]] = []

    holiday_types = {
        "Holiday",
        "Transfer",
        "Bridge",
        "Additional",
    }

    for _, event_row in active_events.iterrows():
        event_locale = str(event_row["locale"])
        locale_name = str(event_row["locale_name"])
        event_type = str(event_row["type"])

        if event_locale == "National":
            applicable_stores = stores_data

        elif event_locale == "Regional":
            applicable_stores = stores_data.loc[
                stores_data["state"].astype(str) == locale_name
            ]

        elif event_locale == "Local":
            applicable_stores = stores_data.loc[
                stores_data["city"].astype(str) == locale_name
            ]

        else:
            continue

        is_holiday = int(event_type in holiday_types)

        for store_number in applicable_stores["store_nbr"]:
            holiday_records.append(
                {
                    "date": event_row["date"],
                    "store_nbr": int(store_number),
                    "is_special_day": int(event_type != "Work Day"),
                    "is_holiday": is_holiday,
                    "is_event": int(event_type == "Event"),
                    "is_work_day": int(event_type == "Work Day"),
                    "is_transfer": int(event_type == "Transfer"),
                    "is_bridge": int(event_type == "Bridge"),
                    "is_additional": int(event_type == "Additional"),
                    "is_national_holiday": int(
                        event_locale == "National" and is_holiday
                    ),
                    "is_regional_holiday": int(
                        event_locale == "Regional" and is_holiday
                    ),
                    "is_local_holiday": int(
                        event_locale == "Local" and is_holiday
                    ),
                }
            )

    if not holiday_records:
        return pd.DataFrame(
            columns=["date", "store_nbr", *HOLIDAY_FEATURE_COLUMNS]
        )

    result = pd.DataFrame(holiday_records)

    result = (
        result.groupby(
            ["date", "store_nbr"],
            as_index=False,
        )[HOLIDAY_FEATURE_COLUMNS]
        .max()
    )

    for feature_column in HOLIDAY_FEATURE_COLUMNS:
        result[feature_column] = result[feature_column].astype("int8")

    result["store_nbr"] = result["store_nbr"].astype("int16")

    return result


def add_horizon_safe_sales_features(
    dataframe: pd.DataFrame,
    forecast_horizon_days: int = FORECAST_HORIZON_DAYS,
    lag_days: Sequence[int] = DEFAULT_LAG_DAYS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add sales history features safe for a direct multi-day forecast.

    Each lag must be at least as long as the forecast horizon. Rolling
    statistics are shifted by the complete forecast horizon.
    """
    invalid_lags = [
        lag_day
        for lag_day in lag_days
        if lag_day < forecast_horizon_days
    ]

    if invalid_lags:
        raise ValueError(
            "Lag days must be greater than or equal to the "
            f"forecast horizon. Invalid values: {invalid_lags}"
        )

    result = dataframe.sort_values(
        ["store_nbr", "family", "date"]
    ).copy()

    sales_groups = result.groupby(
        ["store_nbr", "family"],
        observed=True,
        sort=False,
    )["sales"]

    for lag_day in lag_days:
        result[f"sales_lag_{lag_day}"] = (
            sales_groups.shift(lag_day).astype("float32")
        )

    for rolling_window in rolling_windows:
        mean_column = (
            f"sales_rolling_mean_{rolling_window}"
            f"_shift_{forecast_horizon_days}"
        )

        result[mean_column] = sales_groups.transform(
            lambda sales_series: (
                sales_series.shift(forecast_horizon_days)
                .rolling(
                    window=rolling_window,
                    min_periods=rolling_window,
                )
                .mean()
            )
        ).astype("float32")

    standard_deviation_column = (
        f"sales_rolling_std_28_shift_{forecast_horizon_days}"
    )

    result[standard_deviation_column] = sales_groups.transform(
        lambda sales_series: (
            sales_series.shift(forecast_horizon_days)
            .rolling(
                window=28,
                min_periods=28,
            )
            .std()
        )
    ).astype("float32")

    return result
