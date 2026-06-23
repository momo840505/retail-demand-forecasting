import pandas as pd
import pytest

from retail_forecasting.modeling import (
    CATEGORICAL_FEATURE_COLUMNS,
    align_categorical_features,
    create_inner_validation_split,
    get_model_feature_columns,
)


def test_non_model_columns_are_excluded() -> None:
    test_data = pd.DataFrame(
        {
            "id": [1],
            "date": pd.to_datetime(["2026-01-01"]),
            "sales": [10.0],
            "is_feature_complete": [1],
            "store_nbr": [1],
            "family": ["GROCERY"],
            "sales_lag_16": [8.0],
        }
    )

    feature_columns = get_model_feature_columns(
        test_data
    )

    assert "id" not in feature_columns
    assert "date" not in feature_columns
    assert "sales" not in feature_columns
    assert "is_feature_complete" not in feature_columns

    assert "store_nbr" in feature_columns
    assert "family" in feature_columns
    assert "sales_lag_16" in feature_columns


def test_category_definitions_are_aligned() -> None:
    reference_data = pd.DataFrame(
        {
            "family": ["A", "B"],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
        }
    )

    training_data = reference_data.iloc[[0]].copy()
    validation_data = reference_data.iloc[[1]].copy()

    aligned_training, aligned_validation = (
        align_categorical_features(
            training_features=training_data,
            validation_features=validation_data,
            reference_data=reference_data,
        )
    )

    for column_name in CATEGORICAL_FEATURE_COLUMNS:
        assert (
            str(aligned_training[column_name].dtype)
            == "category"
        )

        assert (
            str(aligned_validation[column_name].dtype)
            == "category"
        )

        assert (
            aligned_training[column_name]
            .cat.categories
            .equals(
                aligned_validation[
                    column_name
                ].cat.categories
            )
        )


def test_inner_validation_split_is_chronological() -> None:
    split = create_inner_validation_split(
        outer_training_end="2017-06-12",
        validation_days=16,
    )

    assert split.inner_validation_start == pd.Timestamp(
        "2017-05-28"
    )

    assert split.inner_validation_end == pd.Timestamp(
        "2017-06-12"
    )

    assert split.inner_training_end == pd.Timestamp(
        "2017-05-27"
    )

    assert (
        split.inner_training_end
        < split.inner_validation_start
    )


def test_invalid_inner_validation_days_are_rejected() -> None:
    with pytest.raises(ValueError):
        create_inner_validation_split(
            outer_training_end="2017-06-12",
            validation_days=0,
        )
