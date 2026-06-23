from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from retail_forecasting.backtesting import (
    BacktestWindow,
    generate_backtest_windows,
)
from retail_forecasting.metrics import evaluate_forecast
from retail_forecasting.modeling import (
    align_categorical_features,
    create_inner_validation_split,
    get_model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_train.parquet"
)

BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "baseline_summary.csv"
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

FORECAST_HORIZON_DAYS = 16
INNER_VALIDATION_DAYS = 16
BACKTEST_FOLD_COUNT = 4
TRAINING_WINDOW_DAYS = 730

MAX_ESTIMATORS = 1500
EARLY_STOPPING_ROUNDS = 75
RANDOM_SEED = 42

MODEL_NAME = "xgboost_log_target_nested"


def load_modeling_data() -> pd.DataFrame:
    """Load and prepare the processed modeling dataset."""
    modeling_data = pd.read_parquet(
        MODELING_DATA_PATH
    )

    modeling_data["date"] = pd.to_datetime(
        modeling_data["date"]
    )

    for column_name in [
        "family",
        "city",
        "state",
        "type",
    ]:
        modeling_data[column_name] = (
            modeling_data[column_name]
            .astype("category")
        )

    return modeling_data


def get_shared_model_parameters() -> dict[str, object]:
    """Return reproducible model parameters shared by both stages."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "max_bin": 256,
        "enable_categorical": True,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }


def build_tuning_model() -> XGBRegressor:
    """Build the model used to select the number of estimators."""
    return XGBRegressor(
        **get_shared_model_parameters(),
        n_estimators=MAX_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


def build_final_model(
    selected_n_estimators: int,
) -> XGBRegressor:
    """Build the final model without using outer validation."""
    if selected_n_estimators <= 0:
        raise ValueError(
            "Selected estimator count must be positive."
        )

    return XGBRegressor(
        **get_shared_model_parameters(),
        n_estimators=selected_n_estimators,
    )


def validate_complete_validation_grid(
    validation_data: pd.DataFrame,
    validation_days: int,
    label: str,
) -> None:
    """Validate a complete date-store-family forecasting grid."""
    expected_row_count = (
        validation_days
        * int(validation_data["store_nbr"].nunique())
        * int(validation_data["family"].nunique())
    )

    if len(validation_data) != expected_row_count:
        raise ValueError(
            f"{label} contains "
            f"{len(validation_data):,} rows; "
            f"expected {expected_row_count:,}."
        )

    if not validation_data[
        "is_feature_complete"
    ].eq(1).all():
        raise ValueError(
            f"{label} contains incomplete features."
        )


def prepare_feature_pair(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: list[str],
    reference_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select and align model features."""
    training_features = training_data[
        feature_columns
    ]

    validation_features = validation_data[
        feature_columns
    ]

    return align_categorical_features(
        training_features=training_features,
        validation_features=validation_features,
        reference_data=reference_data,
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


def run_fold(
    modeling_data: pd.DataFrame,
    feature_columns: list[str],
    backtest_window: BacktestWindow,
) -> tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    """Run nested model selection and outer validation for one fold."""
    fold_start_time = time.perf_counter()

    outer_training_end = (
        backtest_window.validation_start
        - pd.Timedelta(days=1)
    )

    outer_training_start = (
        outer_training_end
        - pd.Timedelta(
            days=TRAINING_WINDOW_DAYS - 1
        )
    )

    inner_split = create_inner_validation_split(
        outer_training_end=outer_training_end,
        validation_days=INNER_VALIDATION_DAYS,
    )

    if (
        inner_split.inner_training_end
        < outer_training_start
    ):
        raise ValueError(
            "The training window is too short for "
            "the requested inner validation period."
        )

    inner_training_data = modeling_data.loc[
        modeling_data["date"].between(
            outer_training_start,
            inner_split.inner_training_end,
        )
        & modeling_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    inner_validation_data = modeling_data.loc[
        modeling_data["date"].between(
            inner_split.inner_validation_start,
            inner_split.inner_validation_end,
        )
        & modeling_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    validate_complete_validation_grid(
        validation_data=inner_validation_data,
        validation_days=INNER_VALIDATION_DAYS,
        label=(
            f"Fold {backtest_window.fold_number} "
            "inner validation"
        ),
    )

    (
        inner_training_features,
        inner_validation_features,
    ) = prepare_feature_pair(
        training_data=inner_training_data,
        validation_data=inner_validation_data,
        feature_columns=feature_columns,
        reference_data=modeling_data,
    )

    inner_training_target = create_log_target(
        inner_training_data
    )

    inner_validation_target = create_log_target(
        inner_validation_data
    )

    inner_training_row_count = int(
        len(inner_training_data)
    )

    inner_validation_row_count = int(
        len(inner_validation_data)
    )

    print()
    print(
        f"Fold {backtest_window.fold_number}: "
        f"{backtest_window.validation_start.date()} "
        f"to {backtest_window.validation_end.date()}"
    )

    print(
        "Inner training period: "
        f"{outer_training_start.date()} to "
        f"{inner_split.inner_training_end.date()}"
    )

    print(
        "Inner validation period: "
        f"{inner_split.inner_validation_start.date()} to "
        f"{inner_split.inner_validation_end.date()}"
    )

    print(
        f"Inner training rows: "
        f"{len(inner_training_data):,}"
    )

    print(
        f"Inner validation rows: "
        f"{len(inner_validation_data):,}"
    )

    tuning_model = build_tuning_model()

    tuning_model.fit(
        inner_training_features,
        inner_training_target,
        eval_set=[
            (
                inner_validation_features,
                inner_validation_target,
            )
        ],
        verbose=False,
    )

    selected_n_estimators = (
        int(tuning_model.best_iteration) + 1
    )

    best_inner_validation_log_rmse = float(
        tuning_model.best_score
    )

    print(
        "Selected estimators from inner validation: "
        f"{selected_n_estimators}"
    )

    print(
        "Best inner validation log-RMSE: "
        f"{best_inner_validation_log_rmse:.6f}"
    )

    del tuning_model
    del inner_training_data
    del inner_validation_data
    del inner_training_features
    del inner_validation_features
    del inner_training_target
    del inner_validation_target

    gc.collect()

    outer_training_data = modeling_data.loc[
        modeling_data["date"].between(
            outer_training_start,
            outer_training_end,
        )
        & modeling_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    outer_validation_data = modeling_data.loc[
        modeling_data["date"].between(
            backtest_window.validation_start,
            backtest_window.validation_end,
        )
        & modeling_data[
            "is_feature_complete"
        ].eq(1)
    ].copy()

    validate_complete_validation_grid(
        validation_data=outer_validation_data,
        validation_days=FORECAST_HORIZON_DAYS,
        label=(
            f"Fold {backtest_window.fold_number} "
            "outer validation"
        ),
    )

    (
        outer_training_features,
        outer_validation_features,
    ) = prepare_feature_pair(
        training_data=outer_training_data,
        validation_data=outer_validation_data,
        feature_columns=feature_columns,
        reference_data=modeling_data,
    )

    outer_training_target = create_log_target(
        outer_training_data
    )

    final_model = build_final_model(
        selected_n_estimators=selected_n_estimators
    )

    final_model.fit(
        outer_training_features,
        outer_training_target,
        verbose=False,
    )

    predicted_log_sales = final_model.predict(
        outer_validation_features
    )

    predicted_sales = np.clip(
        np.expm1(predicted_log_sales),
        a_min=0.0,
        a_max=None,
    )

    actual_sales = outer_validation_data[
        "sales"
    ].to_numpy(dtype=np.float64)

    fold_metrics = evaluate_forecast(
        actual_values=actual_sales,
        predicted_values=predicted_sales,
    )

    elapsed_seconds = (
        time.perf_counter()
        - fold_start_time
    )

    fold_record = {
        "fold_number": (
            backtest_window.fold_number
        ),
        "outer_training_start": (
            outer_training_start
        ),
        "outer_training_end": (
            outer_training_end
        ),
        "inner_training_end": (
            inner_split.inner_training_end
        ),
        "inner_validation_start": (
            inner_split.inner_validation_start
        ),
        "inner_validation_end": (
            inner_split.inner_validation_end
        ),
        "outer_validation_start": (
            backtest_window.validation_start
        ),
        "outer_validation_end": (
            backtest_window.validation_end
        ),
        "inner_training_row_count": (
            inner_training_row_count
        ),
        "inner_validation_row_count": (
            inner_validation_row_count
        ),
        "outer_training_row_count": int(
            len(outer_training_data)
        ),
        "outer_validation_row_count": int(
            len(outer_validation_data)
        ),
        "selected_n_estimators": (
            selected_n_estimators
        ),
        "best_inner_validation_log_rmse": (
            best_inner_validation_log_rmse
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        **fold_metrics,
    }

    booster_importance = (
        final_model.get_booster().get_score(
            importance_type="gain"
        )
    )

    importance_records = [
        {
            "fold_number": (
                backtest_window.fold_number
            ),
            "feature": feature_name,
            "gain": float(
                booster_importance.get(
                    feature_name,
                    0.0,
                )
            ),
        }
        for feature_name in feature_columns
    ]

    print(
        f"Outer training rows: "
        f"{len(outer_training_data):,}"
    )

    print(
        f"Outer validation rows: "
        f"{len(outer_validation_data):,}"
    )

    print(
        f"Outer RMSLE: "
        f"{fold_metrics['rmsle']:.6f}"
    )

    print(
        f"Outer WAPE: "
        f"{fold_metrics['wape_percentage']:.4f}%"
    )

    print(
        f"Elapsed: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )

    del final_model
    del outer_training_data
    del outer_validation_data
    del outer_training_features
    del outer_validation_features
    del outer_training_target

    gc.collect()

    return (
        fold_record,
        actual_sales,
        predicted_sales,
        importance_records,
    )


def create_figures(
    fold_metrics_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    feature_importance_data: pd.DataFrame,
) -> None:
    """Create model comparison and interpretation figures."""
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        fold_metrics_data["fold_number"],
        fold_metrics_data["rmsle"],
        marker="o",
    )

    plt.title(
        "Nested XGBoost RMSLE Across Outer Folds"
    )

    plt.xlabel("Outer Backtest Fold")
    plt.ylabel("RMSLE — Lower Is Better")
    plt.xticks(
        fold_metrics_data["fold_number"]
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "xgboost_nested_rmsle_by_fold.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    ordered_comparison = comparison_data.sort_values(
        "pooled_rmsle",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        ordered_comparison["model"],
        ordered_comparison["pooled_rmsle"],
    )

    plt.title(
        "Nested XGBoost vs Forecasting Baselines"
    )

    plt.xlabel(
        "Pooled RMSLE — Lower Is Better"
    )

    plt.ylabel("Model")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "xgboost_vs_baselines_rmsle.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    main_comparison = ordered_comparison.loc[
        ordered_comparison["model"] != "zero"
    ]

    plt.figure(figsize=(10, 6))

    plt.barh(
        main_comparison["model"],
        main_comparison["pooled_rmsle"],
    )

    plt.title(
        "Nested XGBoost vs Main Forecasting Baselines"
    )

    plt.xlabel(
        "Pooled RMSLE — Lower Is Better"
    )

    plt.ylabel("Model")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "xgboost_vs_main_baselines_rmsle.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    top_importance = (
        feature_importance_data.head(15)
        .sort_values(
            "mean_gain",
            ascending=True,
        )
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_importance["feature"],
        top_importance["mean_gain"],
    )

    plt.title(
        "Top 15 Nested XGBoost Features by Mean Gain"
    )

    plt.xlabel(
        "Mean Gain Across Outer Backtest Folds"
    )

    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "xgboost_feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run nested chronological XGBoost backtesting."""
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("RUNNING NESTED XGBOOST ROLLING BACKTEST")
    print("=" * 70)

    modeling_data = load_modeling_data()

    feature_columns = get_model_feature_columns(
        modeling_data
    )

    final_training_date = (
        modeling_data["date"].max()
    )

    backtest_windows = generate_backtest_windows(
        final_date=final_training_date,
        forecast_horizon_days=(
            FORECAST_HORIZON_DAYS
        ),
        fold_count=BACKTEST_FOLD_COUNT,
    )

    fold_records: list[dict[str, object]] = []
    all_actual_values: list[np.ndarray] = []
    all_predicted_values: list[np.ndarray] = []
    importance_records: list[dict[str, object]] = []

    for backtest_window in backtest_windows:
        (
            fold_record,
            actual_sales,
            predicted_sales,
            fold_importance_records,
        ) = run_fold(
            modeling_data=modeling_data,
            feature_columns=feature_columns,
            backtest_window=backtest_window,
        )

        fold_records.append(fold_record)
        all_actual_values.append(actual_sales)
        all_predicted_values.append(
            predicted_sales
        )
        importance_records.extend(
            fold_importance_records
        )

    fold_metrics_data = pd.DataFrame(
        fold_records
    )

    pooled_actual_values = np.concatenate(
        all_actual_values
    )

    pooled_predicted_values = np.concatenate(
        all_predicted_values
    )

    pooled_metrics = evaluate_forecast(
        actual_values=pooled_actual_values,
        predicted_values=pooled_predicted_values,
    )

    baseline_summary = pd.read_csv(
        BASELINE_SUMMARY_PATH
    )

    best_rmsle_baseline = baseline_summary.loc[
        baseline_summary[
            "pooled_rmsle"
        ].idxmin()
    ]

    best_wape_baseline = baseline_summary.loc[
        baseline_summary[
            "pooled_wape_percentage"
        ].idxmin()
    ]

    rmsle_improvement_percentage = (
        (
            float(
                best_rmsle_baseline[
                    "pooled_rmsle"
                ]
            )
            - pooled_metrics["rmsle"]
        )
        / float(
            best_rmsle_baseline[
                "pooled_rmsle"
            ]
        )
        * 100
    )

    wape_improvement_percentage = (
        (
            float(
                best_wape_baseline[
                    "pooled_wape_percentage"
                ]
            )
            - pooled_metrics[
                "wape_percentage"
            ]
        )
        / float(
            best_wape_baseline[
                "pooled_wape_percentage"
            ]
        )
        * 100
    )

    xgboost_summary = {
        "model": MODEL_NAME,
        "validation_strategy": (
            "nested_chronological_validation"
        ),
        "training_window_days": (
            TRAINING_WINDOW_DAYS
        ),
        "inner_validation_days": (
            INNER_VALIDATION_DAYS
        ),
        "forecast_horizon_days": (
            FORECAST_HORIZON_DAYS
        ),
        "fold_count": BACKTEST_FOLD_COUNT,
        "pooled_mae": pooled_metrics["mae"],
        "pooled_rmse": pooled_metrics["rmse"],
        "pooled_rmsle": pooled_metrics["rmsle"],
        "pooled_wape_percentage": (
            pooled_metrics[
                "wape_percentage"
            ]
        ),
        "mean_fold_rmsle": float(
            fold_metrics_data["rmsle"].mean()
        ),
        "standard_deviation_fold_rmsle": float(
            fold_metrics_data[
                "rmsle"
            ].std(ddof=0)
        ),
        "mean_fold_wape_percentage": float(
            fold_metrics_data[
                "wape_percentage"
            ].mean()
        ),
        "mean_selected_n_estimators": float(
            fold_metrics_data[
                "selected_n_estimators"
            ].mean()
        ),
        "minimum_selected_n_estimators": int(
            fold_metrics_data[
                "selected_n_estimators"
            ].min()
        ),
        "maximum_selected_n_estimators": int(
            fold_metrics_data[
                "selected_n_estimators"
            ].max()
        ),
        "best_rmsle_baseline": str(
            best_rmsle_baseline["model"]
        ),
        "rmsle_improvement_vs_best_baseline_percentage": (
            rmsle_improvement_percentage
        ),
        "best_wape_baseline": str(
            best_wape_baseline["model"]
        ),
        "wape_improvement_vs_best_baseline_percentage": (
            wape_improvement_percentage
        ),
    }

    xgboost_summary_data = pd.DataFrame(
        [xgboost_summary]
    )

    feature_importance_data = (
        pd.DataFrame(importance_records)
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_gain=("gain", "mean"),
            folds_with_nonzero_gain=(
                "gain",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
        )
        .sort_values(
            "mean_gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    comparison_columns = [
        "model",
        "pooled_mae",
        "pooled_rmse",
        "pooled_rmsle",
        "pooled_wape_percentage",
    ]

    comparison_data = pd.concat(
        [
            baseline_summary[
                comparison_columns
            ],
            xgboost_summary_data[
                comparison_columns
            ],
        ],
        ignore_index=True,
    ).sort_values(
        "pooled_rmsle",
        ascending=True,
    )

    fold_metrics_path = (
        REPORT_DIRECTORY
        / "xgboost_metrics_by_fold.csv"
    )

    summary_path = (
        REPORT_DIRECTORY
        / "xgboost_summary.csv"
    )

    importance_path = (
        REPORT_DIRECTORY
        / "xgboost_feature_importance.csv"
    )

    comparison_path = (
        REPORT_DIRECTORY
        / "model_comparison.csv"
    )

    configuration_path = (
        REPORT_DIRECTORY
        / "xgboost_backtest_config.json"
    )

    fold_metrics_data.to_csv(
        fold_metrics_path,
        index=False,
    )

    xgboost_summary_data.to_csv(
        summary_path,
        index=False,
    )

    feature_importance_data.to_csv(
        importance_path,
        index=False,
    )

    comparison_data.to_csv(
        comparison_path,
        index=False,
    )

    configuration = {
        "model_name": MODEL_NAME,
        "validation_strategy": (
            "nested_chronological_validation"
        ),
        "outer_fold_count": (
            BACKTEST_FOLD_COUNT
        ),
        "forecast_horizon_days": (
            FORECAST_HORIZON_DAYS
        ),
        "inner_validation_days": (
            INNER_VALIDATION_DAYS
        ),
        "training_window_days": (
            TRAINING_WINDOW_DAYS
        ),
        "maximum_estimators": (
            MAX_ESTIMATORS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "random_seed": RANDOM_SEED,
        "shared_model_parameters": (
            get_shared_model_parameters()
        ),
        "target_transformation": (
            "log1p"
        ),
        "prediction_inverse_transformation": (
            "clip(expm1(prediction), lower=0)"
        ),
        "outer_validation_used_for_model_selection": False,
    }

    configuration_path.write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    create_figures(
        fold_metrics_data=fold_metrics_data,
        comparison_data=comparison_data,
        feature_importance_data=(
            feature_importance_data
        ),
    )

    print()
    print("=" * 70)
    print("NESTED XGBOOST BACKTEST COMPLETE")
    print("=" * 70)

    print(
        xgboost_summary_data.to_string(
            index=False
        )
    )

    print()
    print("Model comparison:")

    print(
        comparison_data.to_string(
            index=False
        )
    )

    print()
    print(
        f"Fold metrics: {fold_metrics_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Feature importance: "
        f"{importance_path}"
    )

    print(
        f"Comparison: {comparison_path}"
    )

    print(
        f"Configuration: "
        f"{configuration_path}"
    )


if __name__ == "__main__":
    main()
