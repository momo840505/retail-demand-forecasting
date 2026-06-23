from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "final_forecast.csv"
)

MODELING_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_train.parquet"
)

STORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stores.csv"
)

MODEL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "model_comparison.csv"
)

FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "xgboost_metrics_by_fold.csv"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "xgboost_feature_importance.csv"
)

MODEL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "xgboost_summary.csv"
)

DASHBOARD_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "dashboard"
    / "data"
)

RECENT_HISTORY_DAYS = 90


def validate_required_files() -> None:
    """Confirm that all dashboard source files exist."""
    required_files = [
        FORECAST_PATH,
        MODELING_TRAIN_PATH,
        STORES_PATH,
        MODEL_COMPARISON_PATH,
        FOLD_METRICS_PATH,
        FEATURE_IMPORTANCE_PATH,
        MODEL_SUMMARY_PATH,
    ]

    missing_files = [
        str(file_path)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing dashboard source files:\n"
            + "\n".join(missing_files)
        )


def load_store_metadata() -> pd.DataFrame:
    """Load one metadata row for each store."""
    store_data = pd.read_csv(
        STORES_PATH,
        dtype={
            "store_nbr": "int16",
            "city": "string",
            "state": "string",
            "type": "string",
            "cluster": "int16",
        },
    )

    if store_data["store_nbr"].duplicated().any():
        raise ValueError(
            "Duplicate store metadata records were detected."
        )

    return store_data


def prepare_forecast_data(
    store_data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the complete future forecast for the dashboard."""
    forecast_data = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "string",
            "onpromotion": "int32",
            "predicted_sales": "float32",
        },
    )

    if len(forecast_data) != 28_512:
        raise ValueError(
            "Dashboard forecast must contain 28,512 rows."
        )

    if forecast_data["date"].nunique() != 16:
        raise ValueError(
            "Dashboard forecast must contain 16 dates."
        )

    if forecast_data["id"].duplicated().any():
        raise ValueError(
            "Duplicate forecast IDs were detected."
        )

    if forecast_data["predicted_sales"].isna().any():
        raise ValueError(
            "Forecast predictions contain missing values."
        )

    if forecast_data["predicted_sales"].lt(0).any():
        raise ValueError(
            "Forecast predictions contain negative values."
        )

    forecast_data = forecast_data.merge(
        store_data,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )

    return forecast_data.sort_values(
        [
            "date",
            "store_nbr",
            "family",
        ]
    ).reset_index(drop=True)


def prepare_recent_actuals(
    store_data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the final 90 days of actual sales."""
    required_columns = [
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
    ]

    historical_data = pd.read_parquet(
        MODELING_TRAIN_PATH,
        columns=required_columns,
    )

    historical_data["date"] = pd.to_datetime(
        historical_data["date"]
    )

    history_end = historical_data["date"].max()

    history_start = (
        history_end
        - pd.Timedelta(
            days=RECENT_HISTORY_DAYS - 1
        )
    )

    recent_actuals = historical_data.loc[
        historical_data["date"].between(
            history_start,
            history_end,
        )
    ].copy()

    recent_actuals["family"] = (
        recent_actuals["family"].astype("string")
    )

    recent_actuals = recent_actuals.merge(
        store_data,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )

    return recent_actuals.sort_values(
        [
            "date",
            "store_nbr",
            "family",
        ]
    ).reset_index(drop=True)


def main() -> None:
    """Create deployment-ready dashboard datasets."""
    print("=" * 70)
    print("PREPARING DASHBOARD DATA")
    print("=" * 70)

    validate_required_files()

    DASHBOARD_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    store_data = load_store_metadata()

    forecast_data = prepare_forecast_data(
        store_data=store_data,
    )

    recent_actuals = prepare_recent_actuals(
        store_data=store_data,
    )

    model_comparison = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

    fold_metrics = pd.read_csv(
        FOLD_METRICS_PATH,
        parse_dates=[
            "outer_validation_start",
            "outer_validation_end",
        ],
    )

    feature_importance = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    model_summary = pd.read_csv(
        MODEL_SUMMARY_PATH
    )

    forecast_output_path = (
        DASHBOARD_DATA_DIRECTORY
        / "forecast.parquet"
    )

    actuals_output_path = (
        DASHBOARD_DATA_DIRECTORY
        / "recent_actuals.parquet"
    )

    forecast_data.to_parquet(
        forecast_output_path,
        index=False,
        compression="zstd",
    )

    recent_actuals.to_parquet(
        actuals_output_path,
        index=False,
        compression="zstd",
    )

    store_data.to_csv(
        DASHBOARD_DATA_DIRECTORY
        / "stores.csv",
        index=False,
    )

    model_comparison.to_csv(
        DASHBOARD_DATA_DIRECTORY
        / "model_comparison.csv",
        index=False,
    )

    fold_metrics.to_csv(
        DASHBOARD_DATA_DIRECTORY
        / "fold_metrics.csv",
        index=False,
    )

    feature_importance.to_csv(
        DASHBOARD_DATA_DIRECTORY
        / "feature_importance.csv",
        index=False,
    )

    model_summary.to_csv(
        DASHBOARD_DATA_DIRECTORY
        / "model_summary.csv",
        index=False,
    )

    manifest = {
        "forecast_row_count": int(
            len(forecast_data)
        ),
        "forecast_date_start": (
            forecast_data["date"]
            .min()
            .date()
            .isoformat()
        ),
        "forecast_date_end": (
            forecast_data["date"]
            .max()
            .date()
            .isoformat()
        ),
        "forecast_horizon_days": int(
            forecast_data["date"].nunique()
        ),
        "forecast_store_count": int(
            forecast_data["store_nbr"].nunique()
        ),
        "forecast_family_count": int(
            forecast_data["family"].nunique()
        ),
        "recent_actual_row_count": int(
            len(recent_actuals)
        ),
        "recent_actual_date_start": (
            recent_actuals["date"]
            .min()
            .date()
            .isoformat()
        ),
        "recent_actual_date_end": (
            recent_actuals["date"]
            .max()
            .date()
            .isoformat()
        ),
        "source_model": (
            "xgboost_log_target_nested"
        ),
        "performance_source": (
            "nested chronological backtesting"
        ),
        "test_labels_available": False,
    }

    manifest_path = (
        DASHBOARD_DATA_DIRECTORY
        / "dashboard_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Forecast rows: "
        f"{len(forecast_data):,}"
    )

    print(
        f"Recent actual rows: "
        f"{len(recent_actuals):,}"
    )

    print(
        "Forecast period: "
        f"{manifest['forecast_date_start']} "
        f"to "
        f"{manifest['forecast_date_end']}"
    )

    print(
        "Recent actual period: "
        f"{manifest['recent_actual_date_start']} "
        f"to "
        f"{manifest['recent_actual_date_end']}"
    )

    print(
        f"Dashboard data: "
        f"{DASHBOARD_DATA_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
