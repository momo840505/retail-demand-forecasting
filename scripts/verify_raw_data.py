from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "data"

EXPECTED_COLUMNS = {
    "train.csv": [
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
    ],
    "test.csv": [
        "id",
        "date",
        "store_nbr",
        "family",
        "onpromotion",
    ],
    "stores.csv": [
        "store_nbr",
        "city",
        "state",
        "type",
        "cluster",
    ],
    "oil.csv": [
        "date",
        "dcoilwtico",
    ],
    "holidays_events.csv": [
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred",
    ],
    "transactions.csv": [
        "date",
        "store_nbr",
        "transactions",
    ],
    "sample_submission.csv": [
        "id",
        "sales",
    ],
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the whole file into memory."""
    checksum = hashlib.sha256()

    with file_path.open("rb") as input_file:
        for file_chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            checksum.update(file_chunk)

    return checksum.hexdigest()


def count_data_rows(file_path: Path) -> int:
    """Count CSV rows while excluding the header row."""
    with file_path.open("r", encoding="utf-8", newline="") as input_file:
        row_count = sum(1 for _ in csv.reader(input_file))

    return max(row_count - 1, 0)


def validate_raw_data() -> None:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    validation_errors: list[str] = []

    for file_name, expected_columns in EXPECTED_COLUMNS.items():
        file_path = RAW_DATA_DIRECTORY / file_name

        if not file_path.exists():
            validation_errors.append(f"Missing required file: {file_name}")
            continue

        actual_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
        columns_are_valid = actual_columns == expected_columns

        if not columns_are_valid:
            validation_errors.append(
                f"{file_name}: expected columns {expected_columns}, "
                f"but found {actual_columns}"
            )

        manifest_records.append(
            {
                "file_name": file_name,
                "file_size_bytes": file_path.stat().st_size,
                "data_row_count": count_data_rows(file_path),
                "sha256": calculate_sha256(file_path),
                "column_validation": "PASS" if columns_are_valid else "FAIL",
            }
        )

    manifest_path = REPORT_DIRECTORY / "raw_data_manifest.csv"

    pd.DataFrame(manifest_records).sort_values("file_name").to_csv(
        manifest_path,
        index=False,
    )

    validation_result = {
        "status": "PASS" if not validation_errors else "FAIL",
        "raw_data_directory": str(RAW_DATA_DIRECTORY),
        "expected_file_count": len(EXPECTED_COLUMNS),
        "validated_file_count": len(manifest_records),
        "errors": validation_errors,
    }

    result_path = REPORT_DIRECTORY / "raw_data_validation.json"

    result_path.write_text(
        json.dumps(validation_result, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("RAW DATA VALIDATION")
    print("=" * 60)
    print(f"Status: {validation_result['status']}")
    print(f"Validated files: {len(manifest_records)}/{len(EXPECTED_COLUMNS)}")
    print(f"Manifest: {manifest_path}")
    print(f"Validation result: {result_path}")

    if validation_errors:
        print("\nErrors:")

        for validation_error in validation_errors:
            print(f"- {validation_error}")

        raise SystemExit(1)

    print("\nAll required raw data files passed validation.")


if __name__ == "__main__":
    validate_raw_data()
