from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from retail_forecasting.modeling import (
    CATEGORICAL_FEATURE_COLUMNS,
    align_categorical_features,
    create_inner_validation_split,
    get_model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_train.parquet"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_test.parquet"
)

BACKTEST_CONFIG_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "xgboost_backtest_config.json"
)

BACKTEST_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "xgboost_summary.csv"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

PREDICTION_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

MODEL_FILE_NAME = "xgboost_final_model.json"
SUBMISSION_FILE_NAME = "store_sales_submission.csv"
FORECAST_FILE_NAME = "final_forecast.csv"


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum for an output artifact."""
    checksum = hashlib.sha256()

    with file_path.open("rb") as input_file:
        for file_chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            checksum.update(file_chunk)

    return checksum.hexdigest()


def load_configuration() -> dict[str, object]:
    """Load the exact configuration evaluated during backtesting."""
    if not BACKTEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing backtest configuration: {BACKTEST_CONFIG_PATH}"
        )

    return json.loads(
        BACKTEST_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed training and future forecasting datasets."""
    training_data = pd.read_parquet(
        TRAIN_DATA_PATH
    )

    test_data = pd.read_parquet(
        TEST_DATA_PATH
    )

    training_data["date"] = pd.to_datetime(
        training_data["date"]
    )

    test_data["date"] = pd.to_datetime(
        test_data["date"]
    )

    for column_name in CATEGORICAL_FEATURE_COLUMNS:
        training_data[column_name] = (
            training_data[column_name]
            .astype("category")
        )

        test_data[column_name] = (
            test_data[column_name]
            .astype("category")
        )

    return training_data, test_data


def build_tuning_model(
    configuration: dict[str, object],
) -> XGBRegressor:
    """Build the model used only for estimator selection."""
    model_parameters = dict(
        configuration["shared_model_parameters"]
    )

    return XGBRegressor(
        **model_parameters,
        n_estimators=int(
            configuration["maximum_estimators"]
        ),
        early_stopping_rounds=int(
            configuration["early_stopping_rounds"]
        ),
    )


def build_final_model(
    configuration: dict[str, object],
    selected_n_estimators: int,
) -> XGBRegressor:
    """Build the final model using the selected model complexity."""
    if selected_n_estimators <= 0:
        raise ValueError(
            "Selected estimator count must be positive."
        )

    model_parameters = dict(
        configuration["shared_model_parameters"]
    )

    return XGBRegressor(
        **model_parameters,
        n_estimators=selected_n_estimators,
    )


def create_log_target(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Create the log-transformed non-negative sales target."""
    return np.log1p(
        dataframe["sales"].to_numpy(
            dtype=np.float64
        )
    )


def validate_forecast(
    test_data: pd.DataFrame,
    predicted_sales: np.ndarray,
) -> None:
    """Validate final forecast structure and prediction values."""
    if len(test_data) != 28_512:
        raise ValueError(
            "The final forecasting dataset must contain 28,512 rows."
        )

    if test_data["id"].duplicated().any():
        raise ValueError(
            "Duplicate test IDs were detected."
        )

    if test_data["date"].nunique() != 16:
        raise ValueError(
            "The final forecast must cover exactly 16 dates."
        )

    if not test_data[
        "is_feature_complete"
    ].eq(1).all():
        raise ValueError(
            "The final test dataset contains incomplete features."
        )

    if len(predicted_sales) != len(test_data):
        raise ValueError(
            "Prediction count does not match the test row count."
        )

    if not np.isfinite(predicted_sales).all():
        raise ValueError(
            "Final predictions contain missing or infinite values."
        )

    if (predicted_sales < 0).any():
        raise ValueError(
            "Final predictions contain negative sales."
        )


def main() -> None:
    """Train the final model and generate the 16-day forecast."""
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("TRAINING FINAL XGBOOST FORECASTING MODEL")
    print("=" * 70)

    configuration = load_configuration()

    (
        training_data,
        test_data,
    ) = load_modeling_data()

    feature_columns = get_model_feature_columns(
        training_data
    )

    missing_test_features = [
        feature_column
        for feature_column in feature_columns
        if feature_column not in test_data.columns
    ]

    if missing_test_features:
        raise ValueError(
            "The test dataset is missing model features: "
            f"{missing_test_features}"
        )

    training_window_days = int(
        configuration["training_window_days"]
    )

    inner_validation_days = int(
        configuration["inner_validation_days"]
    )

    final_training_end = training_data[
        "date"
    ].max()

    final_training_start = (
        final_training_end
        - pd.Timedelta(
            days=training_window_days - 1
        )
    )

    inner_split = create_inner_validation_split(
        outer_training_end=final_training_end,
        validation_days=inner_validation_days,
    )

    tuning_training_data = training_data.loc[
        training_data["date"].between(
            final_training_start,
            inner_split.inner_training_end,
        )
        & training_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    tuning_validation_data = training_data.loc[
        training_data["date"].between(
            inner_split.inner_validation_start,
            inner_split.inner_validation_end,
        )
        & training_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    final_training_data = training_data.loc[
        training_data["date"].between(
            final_training_start,
            final_training_end,
        )
        & training_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    final_test_data = test_data.loc[
        test_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    expected_validation_rows = (
        inner_validation_days
        * int(
            tuning_validation_data[
                "store_nbr"
            ].nunique()
        )
        * int(
            tuning_validation_data[
                "family"
            ].nunique()
        )
    )

    if (
        len(tuning_validation_data)
        != expected_validation_rows
    ):
        raise ValueError(
            "The final inner validation grid is incomplete."
        )

    category_reference_data = pd.concat(
        [
            training_data[
                CATEGORICAL_FEATURE_COLUMNS
            ],
            test_data[
                CATEGORICAL_FEATURE_COLUMNS
            ],
        ],
        ignore_index=True,
    )

    (
        tuning_training_features,
        tuning_validation_features,
    ) = align_categorical_features(
        training_features=(
            tuning_training_data[
                feature_columns
            ]
        ),
        validation_features=(
            tuning_validation_data[
                feature_columns
            ]
        ),
        reference_data=category_reference_data,
    )

    tuning_training_target = create_log_target(
        tuning_training_data
    )

    tuning_validation_target = create_log_target(
        tuning_validation_data
    )

    print(
        "Final tuning training period: "
        f"{final_training_start.date()} to "
        f"{inner_split.inner_training_end.date()}"
    )

    print(
        "Final tuning validation period: "
        f"{inner_split.inner_validation_start.date()} to "
        f"{inner_split.inner_validation_end.date()}"
    )

    print(
        f"Tuning training rows: "
        f"{len(tuning_training_data):,}"
    )

    print(
        f"Tuning validation rows: "
        f"{len(tuning_validation_data):,}"
    )

    tuning_model = build_tuning_model(
        configuration
    )

    tuning_model.fit(
        tuning_training_features,
        tuning_training_target,
        eval_set=[
            (
                tuning_validation_features,
                tuning_validation_target,
            )
        ],
        verbose=False,
    )

    selected_n_estimators = (
        int(tuning_model.best_iteration) + 1
    )

    best_validation_log_rmse = float(
        tuning_model.best_score
    )

    print(
        "Selected estimators: "
        f"{selected_n_estimators}"
    )

    print(
        "Best tuning validation log-RMSE: "
        f"{best_validation_log_rmse:.6f}"
    )

    del tuning_model
    del tuning_training_features
    del tuning_validation_features
    del tuning_training_target
    del tuning_validation_target

    gc.collect()

    (
        final_training_features,
        final_test_features,
    ) = align_categorical_features(
        training_features=(
            final_training_data[
                feature_columns
            ]
        ),
        validation_features=(
            final_test_data[
                feature_columns
            ]
        ),
        reference_data=category_reference_data,
    )

    final_training_target = create_log_target(
        final_training_data
    )

    print(
        "Final model training period: "
        f"{final_training_start.date()} to "
        f"{final_training_end.date()}"
    )

    print(
        f"Final model training rows: "
        f"{len(final_training_data):,}"
    )

    print(
        f"Forecast rows: "
        f"{len(final_test_data):,}"
    )

    final_model = build_final_model(
        configuration=configuration,
        selected_n_estimators=(
            selected_n_estimators
        ),
    )

    final_model.fit(
        final_training_features,
        final_training_target,
        verbose=False,
    )

    predicted_log_sales = final_model.predict(
        final_test_features
    )

    predicted_sales = np.clip(
        np.expm1(predicted_log_sales),
        a_min=0.0,
        a_max=None,
    )

    validate_forecast(
        test_data=final_test_data,
        predicted_sales=predicted_sales,
    )

    model_path = (
        MODEL_DIRECTORY
        / MODEL_FILE_NAME
    )

    submission_path = (
        PREDICTION_DIRECTORY
        / SUBMISSION_FILE_NAME
    )

    forecast_path = (
        PREDICTION_DIRECTORY
        / FORECAST_FILE_NAME
    )

    final_model.save_model(
        model_path
    )

    submission_data = final_test_data[
        ["id"]
    ].copy()

    submission_data["sales"] = (
        predicted_sales.astype("float64")
    )

    submission_data.to_csv(
        submission_path,
        index=False,
    )

    forecast_data = final_test_data[
        [
            "id",
            "date",
            "store_nbr",
            "family",
            "onpromotion",
        ]
    ].copy()

    forecast_data[
        "predicted_sales"
    ] = predicted_sales.astype("float64")

    forecast_data.to_csv(
        forecast_path,
        index=False,
    )

    forecast_sample_path = (
        REPORT_DIRECTORY
        / "final_forecast_sample.csv"
    )

    forecast_data.head(100).to_csv(
        forecast_sample_path,
        index=False,
    )

    daily_forecast_summary = (
        forecast_data.groupby(
            "date",
            as_index=False,
        )
        .agg(
            total_predicted_sales=(
                "predicted_sales",
                "sum",
            ),
            average_predicted_sales=(
                "predicted_sales",
                "mean",
            ),
            promoted_record_count=(
                "onpromotion",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
        )
        .sort_values("date")
    )

    daily_summary_path = (
        REPORT_DIRECTORY
        / "final_forecast_by_date.csv"
    )

    daily_forecast_summary.to_csv(
        daily_summary_path,
        index=False,
    )

    booster_importance = (
        final_model.get_booster().get_score(
            importance_type="gain"
        )
    )

    feature_importance_data = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": [
                float(
                    booster_importance.get(
                        feature_name,
                        0.0,
                    )
                )
                for feature_name in feature_columns
            ],
        }
    ).sort_values(
        "gain",
        ascending=False,
    )

    importance_path = (
        REPORT_DIRECTORY
        / "final_model_feature_importance.csv"
    )

    feature_importance_data.to_csv(
        importance_path,
        index=False,
    )

    category_levels = {
        column_name: [
            str(category_value)
            for category_value in (
                category_reference_data[
                    column_name
                ]
                .astype("category")
                .cat.categories
                .tolist()
            )
        ]
        for column_name in (
            CATEGORICAL_FEATURE_COLUMNS
        )
    }

    backtest_summary = pd.read_csv(
        BACKTEST_SUMMARY_PATH
    ).iloc[0]

    metadata = {
        "model_name": (
            "xgboost_final_16_day_forecast"
        ),
        "model_file": str(
            model_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "forecast_horizon_days": 16,
        "training_window_days": (
            training_window_days
        ),
        "final_training_start": (
            final_training_start.date().isoformat()
        ),
        "final_training_end": (
            final_training_end.date().isoformat()
        ),
        "tuning_validation_start": (
            inner_split
            .inner_validation_start
            .date()
            .isoformat()
        ),
        "tuning_validation_end": (
            inner_split
            .inner_validation_end
            .date()
            .isoformat()
        ),
        "selected_n_estimators": (
            selected_n_estimators
        ),
        "best_tuning_validation_log_rmse": (
            best_validation_log_rmse
        ),
        "final_training_row_count": int(
            len(final_training_data)
        ),
        "forecast_row_count": int(
            len(forecast_data)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "feature_columns": (
            feature_columns
        ),
        "category_levels": (
            category_levels
        ),
        "prediction_minimum": float(
            predicted_sales.min()
        ),
        "prediction_average": float(
            predicted_sales.mean()
        ),
        "prediction_maximum": float(
            predicted_sales.max()
        ),
        "zero_prediction_count": int(
            (predicted_sales == 0).sum()
        ),
        "formal_backtest_pooled_rmsle": float(
            backtest_summary[
                "pooled_rmsle"
            ]
        ),
        "formal_backtest_pooled_wape_percentage": float(
            backtest_summary[
                "pooled_wape_percentage"
            ]
        ),
        "test_performance_available": False,
        "test_performance_note": (
            "The Kaggle test labels are not available locally. "
            "Reported model performance comes from nested "
            "chronological backtesting only."
        ),
        "model_sha256": (
            calculate_sha256(model_path)
        ),
        "submission_sha256": (
            calculate_sha256(submission_path)
        ),
    }

    metadata_path = (
        REPORT_DIRECTORY
        / "final_model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        daily_forecast_summary["date"],
        daily_forecast_summary[
            "total_predicted_sales"
        ],
        marker="o",
    )

    plt.title(
        "Final 16-Day Total Sales Forecast"
    )

    plt.xlabel("Forecast Date")
    plt.ylabel("Total Predicted Sales")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    forecast_figure_path = (
        FIGURE_DIRECTORY
        / "final_16_day_sales_forecast.png"
    )

    plt.savefig(
        forecast_figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    top_importance = (
        feature_importance_data.head(15)
        .sort_values(
            "gain",
            ascending=True,
        )
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_importance["feature"],
        top_importance["gain"],
    )

    plt.title(
        "Final XGBoost Model Feature Importance"
    )

    plt.xlabel("Gain")
    plt.ylabel("Feature")
    plt.tight_layout()

    final_importance_figure_path = (
        FIGURE_DIRECTORY
        / "final_model_feature_importance.png"
    )

    plt.savefig(
        final_importance_figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 70)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Selected estimators: "
        f"{selected_n_estimators}"
    )

    print(
        f"Final training rows: "
        f"{len(final_training_data):,}"
    )

    print(
        f"Forecast rows: "
        f"{len(forecast_data):,}"
    )

    print(
        f"Forecast period: "
        f"{forecast_data['date'].min().date()} "
        f"to "
        f"{forecast_data['date'].max().date()}"
    )

    print(
        f"Minimum prediction: "
        f"{predicted_sales.min():.4f}"
    )

    print(
        f"Average prediction: "
        f"{predicted_sales.mean():.4f}"
    )

    print(
        f"Maximum prediction: "
        f"{predicted_sales.max():.4f}"
    )

    print(f"Model: {model_path}")
    print(f"Submission: {submission_path}")
    print(f"Forecast: {forecast_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Forecast figure: {forecast_figure_path}")


if __name__ == "__main__":
    main()
