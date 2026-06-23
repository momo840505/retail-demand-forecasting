from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def prepare_forecast_arrays(
    actual_values: Sequence[float],
    predicted_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert forecast values into numerical arrays."""
    actual_array = np.asarray(
        actual_values,
        dtype=np.float64,
    )

    predicted_array = np.asarray(
        predicted_values,
        dtype=np.float64,
    )

    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "Actual and predicted arrays must have identical shapes."
        )

    if actual_array.size == 0:
        raise ValueError(
            "Forecast evaluation requires at least one observation."
        )

    if not np.isfinite(actual_array).all():
        raise ValueError(
            "Actual values contain missing or infinite values."
        )

    if not np.isfinite(predicted_array).all():
        raise ValueError(
            "Predicted values contain missing or infinite values."
        )

    if (actual_array < 0).any():
        raise ValueError(
            "RMSLE cannot be calculated with negative actual values."
        )

    predicted_array = np.clip(
        predicted_array,
        a_min=0.0,
        a_max=None,
    )

    return actual_array, predicted_array


def evaluate_forecast(
    actual_values: Sequence[float],
    predicted_values: Sequence[float],
) -> dict[str, float]:
    """Calculate portfolio forecasting metrics."""
    actual_array, predicted_array = prepare_forecast_arrays(
        actual_values=actual_values,
        predicted_values=predicted_values,
    )

    forecast_errors = predicted_array - actual_array
    absolute_errors = np.abs(forecast_errors)

    mean_absolute_error = float(
        np.mean(absolute_errors)
    )

    root_mean_squared_error = float(
        np.sqrt(
            np.mean(
                np.square(forecast_errors)
            )
        )
    )

    logarithmic_errors = (
        np.log1p(predicted_array)
        - np.log1p(actual_array)
    )

    root_mean_squared_logarithmic_error = float(
        np.sqrt(
            np.mean(
                np.square(logarithmic_errors)
            )
        )
    )

    total_actual_sales = float(
        np.sum(actual_array)
    )

    if total_actual_sales == 0:
        weighted_absolute_percentage_error = float("nan")

    else:
        weighted_absolute_percentage_error = float(
            np.sum(absolute_errors)
            / total_actual_sales
            * 100
        )

    return {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
        "rmsle": root_mean_squared_logarithmic_error,
        "wape_percentage": weighted_absolute_percentage_error,
    }
