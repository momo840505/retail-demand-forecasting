from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestWindow:
    """A single chronological forecasting validation window."""

    fold_number: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    @property
    def forecast_origin(self) -> pd.Timestamp:
        """Return the final date available before forecasting."""
        return self.validation_start - pd.Timedelta(days=1)


def generate_backtest_windows(
    final_date: pd.Timestamp | str,
    forecast_horizon_days: int,
    fold_count: int,
) -> list[BacktestWindow]:
    """Generate non-overlapping chronological validation windows."""
    if forecast_horizon_days <= 0:
        raise ValueError(
            "Forecast horizon must be greater than zero."
        )

    if fold_count <= 0:
        raise ValueError(
            "Fold count must be greater than zero."
        )

    final_timestamp = pd.Timestamp(final_date)

    backtest_windows: list[BacktestWindow] = []

    for fold_index in range(fold_count):
        periods_after_current_fold = (
            fold_count - fold_index - 1
        )

        validation_end = (
            final_timestamp
            - pd.Timedelta(
                days=(
                    periods_after_current_fold
                    * forecast_horizon_days
                )
            )
        )

        validation_start = (
            validation_end
            - pd.Timedelta(
                days=forecast_horizon_days - 1
            )
        )

        backtest_windows.append(
            BacktestWindow(
                fold_number=fold_index + 1,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )

    return backtest_windows
