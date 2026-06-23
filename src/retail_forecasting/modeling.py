from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NON_FEATURE_COLUMNS = {
    "id",
    "date",
    "sales",
    "is_feature_complete",
}

CATEGORICAL_FEATURE_COLUMNS = [
    "family",
    "city",
    "state",
    "type",
]


@dataclass(frozen=True)
class InnerValidationSplit:
    """Chronological split used only for model selection."""

    inner_training_end: pd.Timestamp
    inner_validation_start: pd.Timestamp
    inner_validation_end: pd.Timestamp


def create_inner_validation_split(
    outer_training_end: pd.Timestamp | str,
    validation_days: int,
) -> InnerValidationSplit:
    """Create an inner validation period ending at outer training end."""
    if validation_days <= 0:
        raise ValueError(
            "Inner validation days must be greater than zero."
        )

    inner_validation_end = pd.Timestamp(
        outer_training_end
    )

    inner_validation_start = (
        inner_validation_end
        - pd.Timedelta(days=validation_days - 1)
    )

    inner_training_end = (
        inner_validation_start
        - pd.Timedelta(days=1)
    )

    return InnerValidationSplit(
        inner_training_end=inner_training_end,
        inner_validation_start=inner_validation_start,
        inner_validation_end=inner_validation_end,
    )


def get_model_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return model features excluding IDs, dates, and targets."""
    return [
        column_name
        for column_name in dataframe.columns
        if column_name not in NON_FEATURE_COLUMNS
    ]


def align_categorical_features(
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    reference_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply identical category definitions to two datasets."""
    training_result = training_features.copy()
    validation_result = validation_features.copy()

    for column_name in CATEGORICAL_FEATURE_COLUMNS:
        categories = (
            reference_data[column_name]
            .astype("category")
            .cat.categories
        )

        training_result[column_name] = pd.Categorical(
            training_result[column_name],
            categories=categories,
        )

        validation_result[column_name] = pd.Categorical(
            validation_result[column_name],
            categories=categories,
        )

    return training_result, validation_result
