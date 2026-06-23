from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail_forecasting.backtesting import (
    BacktestWindow,
    generate_backtest_windows,
)
from retail_forecasting.baselines import (
    add_weekly_seasonal_naive_prediction,
)
from retail_forecasting.metrics import (
    evaluate_forecast,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_train.parquet"
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
BACKTEST_FOLD_COUNT = 4

BASELINE_PREDICTION_COLUMNS = {
    "zero": "zero_prediction",
    "lag_16": "sales_lag_16",
    "weekly_seasonal_naive": (
        "weekly_seasonal_naive_prediction"
    ),
    "shifted_28_day_mean": (
        "sales_rolling_mean_28_shift_16"
    ),
    "lag_364": "sales_lag_364",
}


def load_modeling_data() -> pd.DataFrame:
    """Load columns required for baseline evaluation."""
    required_columns = [
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "sales_lag_16",
        "sales_lag_364",
        "sales_rolling_mean_28_shift_16",
        "is_feature_complete",
    ]

    modeling_data = pd.read_parquet(
        MODELING_DATA_PATH,
        columns=required_columns,
    )

    modeling_data["date"] = pd.to_datetime(
        modeling_data["date"]
    )

    return modeling_data


def validate_fold(
    validation_data: pd.DataFrame,
    backtest_window: BacktestWindow,
) -> None:
    """Confirm that each fold contains a complete forecast grid."""
    expected_row_count = (
        FORECAST_HORIZON_DAYS
        * int(validation_data["store_nbr"].nunique())
        * int(validation_data["family"].nunique())
    )

    if len(validation_data) != expected_row_count:
        raise ValueError(
            f"Fold {backtest_window.fold_number} has "
            f"{len(validation_data):,} rows; "
            f"expected {expected_row_count:,}."
        )

    if not validation_data[
        "is_feature_complete"
    ].eq(1).all():
        raise ValueError(
            f"Fold {backtest_window.fold_number} contains "
            "incomplete history features."
        )


def add_baseline_predictions(
    full_data: pd.DataFrame,
    backtest_window: BacktestWindow,
) -> pd.DataFrame:
    """Create all baseline forecasts for one validation fold."""
    validation_data = full_data.loc[
        full_data["date"].between(
            backtest_window.validation_start,
            backtest_window.validation_end,
        )
    ].copy()

    validate_fold(
        validation_data=validation_data,
        backtest_window=backtest_window,
    )

    seasonal_predictions = (
        add_weekly_seasonal_naive_prediction(
            full_data=full_data[
                [
                    "id",
                    "date",
                    "store_nbr",
                    "family",
                    "sales",
                ]
            ],
            validation_start=(
                backtest_window.validation_start
            ),
            validation_end=(
                backtest_window.validation_end
            ),
        )
        [
            [
                "id",
                "weekly_seasonal_naive_prediction",
            ]
        ]
    )

    validation_data = validation_data.merge(
        seasonal_predictions,
        on="id",
        how="left",
        validate="one_to_one",
    )

    validation_data["zero_prediction"] = np.float32(
        0.0
    )

    for prediction_column in (
        BASELINE_PREDICTION_COLUMNS.values()
    ):
        if validation_data[
            prediction_column
        ].isna().any():
            raise ValueError(
                f"Missing values detected in "
                f"{prediction_column}."
            )

        validation_data[prediction_column] = (
            validation_data[prediction_column]
            .clip(lower=0)
            .astype("float32")
        )

    validation_data["fold_number"] = (
        backtest_window.fold_number
    )

    validation_data["validation_start"] = (
        backtest_window.validation_start
    )

    validation_data["validation_end"] = (
        backtest_window.validation_end
    )

    validation_data["forecast_origin"] = (
        backtest_window.forecast_origin
    )

    return validation_data


def evaluate_fold_predictions(
    validation_data: pd.DataFrame,
) -> list[dict[str, object]]:
    """Calculate overall metrics for every baseline in a fold."""
    metric_records: list[dict[str, object]] = []

    for model_name, prediction_column in (
        BASELINE_PREDICTION_COLUMNS.items()
    ):
        metrics = evaluate_forecast(
            actual_values=validation_data["sales"],
            predicted_values=(
                validation_data[prediction_column]
            ),
        )

        metric_records.append(
            {
                "fold_number": int(
                    validation_data[
                        "fold_number"
                    ].iloc[0]
                ),
                "validation_start": (
                    validation_data[
                        "validation_start"
                    ].iloc[0]
                ),
                "validation_end": (
                    validation_data[
                        "validation_end"
                    ].iloc[0]
                ),
                "forecast_origin": (
                    validation_data[
                        "forecast_origin"
                    ].iloc[0]
                ),
                "model": model_name,
                "observation_count": int(
                    len(validation_data)
                ),
                **metrics,
            }
        )

    return metric_records


def create_summary_table(
    all_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create pooled and across-fold baseline metrics."""
    summary_records: list[dict[str, object]] = []

    for model_name, prediction_column in (
        BASELINE_PREDICTION_COLUMNS.items()
    ):
        pooled_metrics = evaluate_forecast(
            actual_values=all_predictions["sales"],
            predicted_values=(
                all_predictions[prediction_column]
            ),
        )

        model_fold_metrics = fold_metrics.loc[
            fold_metrics["model"] == model_name
        ]

        summary_records.append(
            {
                "model": model_name,
                "pooled_mae": pooled_metrics["mae"],
                "pooled_rmse": pooled_metrics["rmse"],
                "pooled_rmsle": pooled_metrics["rmsle"],
                "pooled_wape_percentage": (
                    pooled_metrics[
                        "wape_percentage"
                    ]
                ),
                "mean_fold_rmsle": float(
                    model_fold_metrics[
                        "rmsle"
                    ].mean()
                ),
                "standard_deviation_fold_rmsle": float(
                    model_fold_metrics[
                        "rmsle"
                    ].std(ddof=0)
                ),
                "mean_fold_wape_percentage": float(
                    model_fold_metrics[
                        "wape_percentage"
                    ].mean()
                ),
            }
        )

    return (
        pd.DataFrame(summary_records)
        .sort_values(
            "pooled_rmsle",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def create_family_metrics(
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every baseline separately by product family."""
    family_metric_records: list[dict[str, object]] = []

    for family_name, family_data in (
        all_predictions.groupby(
            "family",
            observed=True,
        )
    ):
        for model_name, prediction_column in (
            BASELINE_PREDICTION_COLUMNS.items()
        ):
            metrics = evaluate_forecast(
                actual_values=family_data["sales"],
                predicted_values=(
                    family_data[prediction_column]
                ),
            )

            family_metric_records.append(
                {
                    "family": str(family_name),
                    "model": model_name,
                    "observation_count": int(
                        len(family_data)
                    ),
                    **metrics,
                }
            )

    return (
        pd.DataFrame(family_metric_records)
        .sort_values(
            [
                "family",
                "rmsle",
            ]
        )
        .reset_index(drop=True)
    )


def create_figures(
    fold_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
) -> None:
    """Create baseline performance figures."""
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_rmsle_table = fold_metrics.pivot(
        index="fold_number",
        columns="model",
        values="rmsle",
    )

    plot_axis = fold_rmsle_table.plot(
        figsize=(11, 6),
        marker="o",
    )

    plot_axis.set_title(
        "Baseline RMSLE Across 16-Day Backtest Folds"
    )

    plot_axis.set_xlabel(
        "Backtest Fold"
    )

    plot_axis.set_ylabel(
        "RMSLE — Lower Is Better"
    )

    plot_axis.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "baseline_rmsle_by_fold.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    ordered_summary = summary_metrics.sort_values(
        "pooled_rmsle",
        ascending=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        ordered_summary["model"],
        ordered_summary["pooled_rmsle"],
    )

    plt.title(
        "Pooled Baseline RMSLE"
    )

    plt.xlabel(
        "RMSLE — Lower Is Better"
    )

    plt.ylabel(
        "Baseline Model"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "baseline_pooled_rmsle.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run chronological baseline backtesting."""
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("RUNNING BASELINE ROLLING BACKTEST")
    print("=" * 70)

    modeling_data = load_modeling_data()

    final_training_date = modeling_data["date"].max()

    backtest_windows = generate_backtest_windows(
        final_date=final_training_date,
        forecast_horizon_days=(
            FORECAST_HORIZON_DAYS
        ),
        fold_count=BACKTEST_FOLD_COUNT,
    )

    all_fold_predictions: list[pd.DataFrame] = []
    fold_metric_records: list[dict[str, object]] = []

    for backtest_window in backtest_windows:
        print()
        print(
            f"Fold {backtest_window.fold_number}: "
            f"{backtest_window.validation_start.date()} "
            f"to "
            f"{backtest_window.validation_end.date()}"
        )

        validation_predictions = (
            add_baseline_predictions(
                full_data=modeling_data,
                backtest_window=backtest_window,
            )
        )

        all_fold_predictions.append(
            validation_predictions
        )

        fold_metric_records.extend(
            evaluate_fold_predictions(
                validation_predictions
            )
        )

        print(
            f"Validated rows: "
            f"{len(validation_predictions):,}"
        )

    all_predictions = pd.concat(
        all_fold_predictions,
        ignore_index=True,
    )

    fold_metrics = pd.DataFrame(
        fold_metric_records
    )

    summary_metrics = create_summary_table(
        all_predictions=all_predictions,
        fold_metrics=fold_metrics,
    )

    family_metrics = create_family_metrics(
        all_predictions=all_predictions,
    )

    fold_metrics_path = (
        REPORT_DIRECTORY
        / "baseline_metrics_by_fold.csv"
    )

    summary_metrics_path = (
        REPORT_DIRECTORY
        / "baseline_summary.csv"
    )

    family_metrics_path = (
        REPORT_DIRECTORY
        / "baseline_metrics_by_family.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    summary_metrics.to_csv(
        summary_metrics_path,
        index=False,
    )

    family_metrics.to_csv(
        family_metrics_path,
        index=False,
    )

    create_figures(
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
    )

    print()
    print("=" * 70)
    print("BASELINE BACKTEST COMPLETE")
    print("=" * 70)

    print(
        summary_metrics[
            [
                "model",
                "pooled_rmsle",
                "pooled_mae",
                "pooled_rmse",
                "pooled_wape_percentage",
            ]
        ].to_string(
            index=False,
        )
    )

    print()
    print(
        f"Fold metrics: {fold_metrics_path}"
    )

    print(
        f"Summary metrics: {summary_metrics_path}"
    )

    print(
        f"Family metrics: {family_metrics_path}"
    )

    print(
        f"Figures: {FIGURE_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
