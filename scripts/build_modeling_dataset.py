from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecasting.features import (
    FORECAST_HORIZON_DAYS,
    HOLIDAY_FEATURE_COLUMNS,
    add_calendar_features,
    add_horizon_safe_sales_features,
    build_daily_oil_features,
    build_store_holiday_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "data"


def load_source_data() -> dict[str, pd.DataFrame]:
    """Load source datasets with memory-conscious data types."""
    print("Loading train.csv...")

    train_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "train.csv",
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "category",
            "sales": "float32",
            "onpromotion": "int32",
        },
    )

    print("Loading test.csv...")

    test_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "test.csv",
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "category",
            "onpromotion": "int32",
        },
    )

    print("Loading stores.csv...")

    stores_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "stores.csv",
        dtype={
            "store_nbr": "int16",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "int16",
        },
    )

    print("Loading oil.csv...")

    oil_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "oil.csv",
        parse_dates=["date"],
        dtype={
            "dcoilwtico": "float32",
        },
    )

    print("Loading holidays_events.csv...")

    holidays_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "holidays_events.csv",
        parse_dates=["date"],
    )

    return {
        "train": train_data,
        "test": test_data,
        "stores": stores_data,
        "oil": oil_data,
        "holidays": holidays_data,
    }


def validate_source_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Validate basic source-data requirements before processing."""
    if train_data.empty:
        raise ValueError("Training data is empty.")

    if test_data.empty:
        raise ValueError("Test data is empty.")

    if train_data["id"].duplicated().any():
        raise ValueError("Duplicate training IDs were detected.")

    if test_data["id"].duplicated().any():
        raise ValueError("Duplicate test IDs were detected.")

    if test_data["date"].nunique() != FORECAST_HORIZON_DAYS:
        raise ValueError(
            "Unexpected forecast horizon. "
            f"Expected {FORECAST_HORIZON_DAYS} dates, "
            f"but found {test_data['date'].nunique()}."
        )

    if test_data["date"].min() <= train_data["date"].max():
        raise ValueError(
            "Test dates must begin after the final training date."
        )


def build_modeling_datasets(
    source_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build leakage-aware training and test modeling datasets."""
    train_data = source_data["train"].copy()
    test_data = source_data["test"].copy()
    stores_data = source_data["stores"].copy()

    validate_source_data(
        train_data=train_data,
        test_data=test_data,
    )

    print("Combining train and test data...")

    train_data["dataset_split"] = "train"
    test_data["dataset_split"] = "test"

    test_data["sales"] = np.nan
    test_data["sales"] = test_data["sales"].astype("float32")

    combined_data = pd.concat(
        [train_data, test_data],
        ignore_index=True,
        sort=False,
    )

    combined_data["family"] = combined_data["family"].astype(
        "category"
    )

    print("Adding store metadata...")

    combined_data = combined_data.merge(
        stores_data,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )

    store_metadata_columns = [
        "city",
        "state",
        "type",
        "cluster",
    ]

    if combined_data[store_metadata_columns].isna().any().any():
        raise ValueError(
            "Missing store metadata was detected after merging."
        )

    print("Adding calendar features...")

    combined_data = add_calendar_features(combined_data)

    print("Adding horizon-safe oil features...")

    oil_features = build_daily_oil_features(
        oil_data=source_data["oil"],
        start_date=combined_data["date"].min(),
        end_date=combined_data["date"].max(),
        forecast_horizon_days=FORECAST_HORIZON_DAYS,
    )

    combined_data = combined_data.merge(
        oil_features,
        on="date",
        how="left",
        validate="many_to_one",
    )

    print("Mapping holidays to applicable stores...")

    holiday_features = build_store_holiday_features(
        stores_data=stores_data,
        holidays_data=source_data["holidays"],
    )

    combined_data = combined_data.merge(
        holiday_features,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )

    for holiday_column in HOLIDAY_FEATURE_COLUMNS:
        combined_data[holiday_column] = (
            combined_data[holiday_column]
            .fillna(0)
            .astype("int8")
        )

    print("Creating horizon-safe sales lags and rolling features...")
    print("This is the slowest step and may take several minutes.")

    combined_data = add_horizon_safe_sales_features(
        dataframe=combined_data,
        forecast_horizon_days=FORECAST_HORIZON_DAYS,
    )

    history_feature_columns = [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_lag_35",
        "sales_lag_364",
        "sales_rolling_mean_7_shift_16",
        "sales_rolling_mean_28_shift_16",
        "sales_rolling_std_28_shift_16",
    ]

    combined_data["is_feature_complete"] = (
        combined_data[history_feature_columns]
        .notna()
        .all(axis=1)
        .astype("int8")
    )

    print("Separating modeling train and test datasets...")

    modeling_train_data = combined_data.loc[
        combined_data["dataset_split"] == "train"
    ].copy()

    modeling_test_data = combined_data.loc[
        combined_data["dataset_split"] == "test"
    ].copy()

    modeling_train_data = modeling_train_data.drop(
        columns=["dataset_split"]
    )

    modeling_test_data = modeling_test_data.drop(
        columns=["dataset_split", "sales"]
    )

    modeling_train_data = modeling_train_data.sort_values(
        "id"
    ).reset_index(drop=True)

    modeling_test_data = modeling_test_data.sort_values(
        "id"
    ).reset_index(drop=True)

    test_history_missing_counts = {
        feature_column: int(
            modeling_test_data[feature_column].isna().sum()
        )
        for feature_column in history_feature_columns
    }

    incomplete_test_features = {
        feature_column: missing_count
        for feature_column, missing_count
        in test_history_missing_counts.items()
        if missing_count > 0
    }

    if incomplete_test_features:
        raise ValueError(
            "The test dataset contains missing history features: "
            f"{incomplete_test_features}"
        )

    expected_test_rows = (
        FORECAST_HORIZON_DAYS
        * int(test_data["store_nbr"].nunique())
        * int(test_data["family"].nunique())
    )

    if len(modeling_test_data) != expected_test_rows:
        raise ValueError(
            "The generated test dataset is not a complete "
            "date-store-family grid."
        )

    excluded_columns = {
        "id",
        "sales",
    }

    model_feature_columns = [
        column_name
        for column_name in modeling_train_data.columns
        if column_name not in excluded_columns
    ]

    categorical_columns = [
        column_name
        for column_name in model_feature_columns
        if str(modeling_train_data[column_name].dtype) == "category"
        or modeling_train_data[column_name].dtype == "object"
    ]

    modeling_profile = {
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "training_date_start": (
            modeling_train_data["date"].min().date().isoformat()
        ),
        "training_date_end": (
            modeling_train_data["date"].max().date().isoformat()
        ),
        "test_date_start": (
            modeling_test_data["date"].min().date().isoformat()
        ),
        "test_date_end": (
            modeling_test_data["date"].max().date().isoformat()
        ),
        "train_row_count": int(len(modeling_train_data)),
        "test_row_count": int(len(modeling_test_data)),
        "expected_test_row_count": int(expected_test_rows),
        "unique_train_stores": int(
            modeling_train_data["store_nbr"].nunique()
        ),
        "unique_test_stores": int(
            modeling_test_data["store_nbr"].nunique()
        ),
        "unique_train_families": int(
            modeling_train_data["family"].nunique()
        ),
        "unique_test_families": int(
            modeling_test_data["family"].nunique()
        ),
        "train_feature_complete_rows": int(
            modeling_train_data["is_feature_complete"].sum()
        ),
        "test_feature_complete_rows": int(
            modeling_test_data["is_feature_complete"].sum()
        ),
        "test_feature_complete_percentage": round(
            float(
                modeling_test_data["is_feature_complete"].mean()
                * 100
            ),
            4,
        ),
        "model_feature_count": int(len(model_feature_columns)),
        "model_feature_columns": model_feature_columns,
        "categorical_columns": categorical_columns,
        "history_feature_columns": history_feature_columns,
        "test_history_missing_counts": (
            test_history_missing_counts
        ),
        "transactions_used_as_model_feature": False,
        "transactions_exclusion_reason": (
            "Future transaction totals are unavailable during the "
            "forecast period and would create a training-serving "
            "mismatch."
        ),
        "oil_feature_policy": (
            "Oil-price features are shifted by the complete "
            "16-day forecast horizon."
        ),
        "sales_feature_policy": (
            "Sales lags are at least 16 days old and rolling "
            "statistics are shifted by 16 days."
        ),
    }

    return (
        modeling_train_data,
        modeling_test_data,
        modeling_profile,
    )


def main() -> None:
    """Generate processed modeling datasets and validation metadata."""
    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("BUILDING LEAKAGE-AWARE MODELING DATASETS")
    print("=" * 70)

    source_data = load_source_data()

    (
        modeling_train_data,
        modeling_test_data,
        modeling_profile,
    ) = build_modeling_datasets(source_data)

    train_output_path = (
        PROCESSED_DATA_DIRECTORY / "modeling_train.parquet"
    )

    test_output_path = (
        PROCESSED_DATA_DIRECTORY / "modeling_test.parquet"
    )

    profile_output_path = (
        REPORT_DIRECTORY / "modeling_dataset_profile.json"
    )

    print("Writing compressed training parquet...")

    modeling_train_data.to_parquet(
        train_output_path,
        index=False,
        compression="zstd",
    )

    print("Writing compressed test parquet...")

    modeling_test_data.to_parquet(
        test_output_path,
        index=False,
        compression="zstd",
    )

    profile_output_path.write_text(
        json.dumps(
            modeling_profile,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_size_megabytes = round(
        train_output_path.stat().st_size / 1024 / 1024,
        2,
    )

    test_size_megabytes = round(
        test_output_path.stat().st_size / 1024 / 1024,
        2,
    )

    print()
    print("=" * 70)
    print("MODELING DATASET BUILD COMPLETE")
    print("=" * 70)
    print(
        f"Training rows: "
        f"{modeling_profile['train_row_count']:,}"
    )
    print(
        f"Test rows: "
        f"{modeling_profile['test_row_count']:,}"
    )
    print(
        "Complete test feature rows: "
        f"{modeling_profile['test_feature_complete_rows']:,}"
    )
    print(
        "Complete test feature percentage: "
        f"{modeling_profile['test_feature_complete_percentage']}%"
    )
    print(
        f"Model features: "
        f"{modeling_profile['model_feature_count']}"
    )
    print(
        f"Training parquet size: "
        f"{train_size_megabytes} MB"
    )
    print(
        f"Test parquet size: "
        f"{test_size_megabytes} MB"
    )
    print(f"Train output: {train_output_path}")
    print(f"Test output: {test_output_path}")
    print(f"Profile output: {profile_output_path}")
    print("Transactions excluded from model features: True")


if __name__ == "__main__":
    main()
