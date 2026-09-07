import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_modeling_profile_matches_final_model_features() -> None:
    profile = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "data"
            / "modeling_dataset_profile.json"
        ).read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "modeling"
            / "final_model_metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert profile["model_feature_count"] == len(
        profile["model_feature_columns"]
    )
    assert metadata["feature_count"] == len(
        metadata["feature_columns"]
    )
    assert (
        profile["model_feature_columns"]
        == metadata["feature_columns"]
    )


def test_generated_metadata_uses_portable_paths() -> None:
    validation = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "data"
            / "raw_data_validation.json"
        ).read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "modeling"
            / "final_model_metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert validation["raw_data_directory"] == "data/raw"
    assert "\\" not in metadata["model_file"]
