from __future__ import annotations

import pandas as pd


SERIES_KEY_COLUMNS = [
    "store_nbr",
    "family",
]


def add_weekly_seasonal_naive_prediction(
    full_data: pd.DataFrame,
    validation_start: pd.Timestamp | str,
    validation_end: pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Repeat the final seven observed days across the forecast horizon.

    Every source observation occurs strictly before the validation
    period, preventing the baseline from using realised future sales.
    """
    result_data = full_data.copy()
    result_data["date"] = pd.to_datetime(
        result_data["date"]
    )

    validation_start_timestamp = pd.Timestamp(
        validation_start
    )

    validation_end_timestamp = pd.Timestamp(
        validation_end
    )

    validation_data = result_data.loc[
        result_data["date"].between(
            validation_start_timestamp,
            validation_end_timestamp,
        )
    ].copy()

    if validation_data.empty:
        raise ValueError(
            "The requested validation period contains no rows."
        )

    source_period_start = (
        validation_start_timestamp
        - pd.Timedelta(days=7)
    )

    source_period_end = (
        validation_start_timestamp
        - pd.Timedelta(days=1)
    )

    source_data = result_data.loc[
        result_data["date"].between(
            source_period_start,
            source_period_end,
        ),
        [
            *SERIES_KEY_COLUMNS,
            "date",
            "sales",
        ],
    ].copy()

    duplicate_source_keys = source_data.duplicated(
        subset=[
            *SERIES_KEY_COLUMNS,
            "date",
        ]
    )

    if duplicate_source_keys.any():
        raise ValueError(
            "Duplicate source observations were detected."
        )

    validation_data["forecast_day_index"] = (
        validation_data["date"]
        - validation_start_timestamp
    ).dt.days

    validation_data["seasonal_source_date"] = (
        source_period_start
        + pd.to_timedelta(
            validation_data["forecast_day_index"] % 7,
            unit="D",
        )
    )

    source_lookup = source_data.rename(
        columns={
            "date": "seasonal_source_date",
            "sales": "weekly_seasonal_naive_prediction",
        }
    )

    validation_data = validation_data.merge(
        source_lookup,
        on=[
            *SERIES_KEY_COLUMNS,
            "seasonal_source_date",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_prediction_count = int(
        validation_data[
            "weekly_seasonal_naive_prediction"
        ].isna().sum()
    )

    if missing_prediction_count > 0:
        raise ValueError(
            "Weekly seasonal predictions contain "
            f"{missing_prediction_count} missing values."
        )

    if (
        validation_data["seasonal_source_date"]
        >= validation_start_timestamp
    ).any():
        raise ValueError(
            "Seasonal baseline used data from the validation period."
        )

    validation_data[
        "weekly_seasonal_naive_prediction"
    ] = (
        validation_data[
            "weekly_seasonal_naive_prediction"
        ]
        .clip(lower=0)
        .astype("float32")
    )

    return validation_data
