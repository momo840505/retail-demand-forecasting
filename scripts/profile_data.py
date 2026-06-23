from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DATA_REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "data"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"


def calculate_percentage(part: int | float, total: int | float) -> float:
    """Calculate a percentage while safely handling a zero denominator."""
    if total == 0:
        return 0.0

    return round(float(part) / float(total) * 100, 4)


def get_missing_value_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return missing-value counts for every column."""
    return {
        str(column_name): int(dataframe[column_name].isna().sum())
        for column_name in dataframe.columns
    }


def load_datasets() -> dict[str, pd.DataFrame]:
    """Load all raw datasets with memory-conscious data types."""
    train_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "train.csv",
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "category",
            "sales": "float32",
            "onpromotion": "int32",
        },
    )

    test_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "test.csv",
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "category",
            "onpromotion": "int32",
        },
    )

    stores_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "stores.csv",
        dtype={
            "store_nbr": "int16",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "int16",
        },
    )

    oil_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "oil.csv",
        parse_dates=["date"],
        dtype={"dcoilwtico": "float32"},
    )

    holidays_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "holidays_events.csv",
        parse_dates=["date"],
    )

    transactions_data = pd.read_csv(
        RAW_DATA_DIRECTORY / "transactions.csv",
        parse_dates=["date"],
        dtype={
            "store_nbr": "int16",
            "transactions": "int32",
        },
    )

    return {
        "train": train_data,
        "test": test_data,
        "stores": stores_data,
        "oil": oil_data,
        "holidays": holidays_data,
        "transactions": transactions_data,
    }


def create_summary_tables(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reusable sales and promotion summary tables."""
    daily_sales_summary = (
        train_data.groupby("date", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_sales=("sales", "mean"),
            total_onpromotion=("onpromotion", "sum"),
            positive_sales_series=("sales", lambda values: int((values > 0).sum())),
        )
        .sort_values("date")
    )

    family_sales_summary = (
        train_data.groupby("family", observed=True, as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_sales=("sales", "mean"),
            median_sales=("sales", "median"),
            record_count=("sales", "size"),
            zero_sales_count=("sales", lambda values: int((values == 0).sum())),
        )
        .sort_values("total_sales", ascending=False)
    )

    family_sales_summary["family"] = (
        family_sales_summary["family"].astype(str)
    )

    promotion_flag = train_data["onpromotion"].gt(0).rename(
        "is_on_promotion"
    )

    promotion_sales_summary = (
        train_data.groupby(promotion_flag)
        .agg(
            average_sales=("sales", "mean"),
            median_sales=("sales", "median"),
            total_sales=("sales", "sum"),
            record_count=("sales", "size"),
        )
        .reset_index()
    )

    promotion_sales_summary["promotion_status"] = (
        promotion_sales_summary["is_on_promotion"].map(
            {
                True: "On promotion",
                False: "Not on promotion",
            }
        )
    )

    return (
        daily_sales_summary,
        family_sales_summary,
        promotion_sales_summary,
    )


def create_figures(
    daily_sales_summary: pd.DataFrame,
    family_sales_summary: pd.DataFrame,
    promotion_sales_summary: pd.DataFrame,
) -> None:
    """Create portfolio-ready exploratory figures."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 5))
    plt.plot(
        daily_sales_summary["date"],
        daily_sales_summary["total_sales"],
    )
    plt.title("Total Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "daily_total_sales.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    top_family_sales = (
        family_sales_summary.head(15)
        .sort_values("total_sales", ascending=True)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_family_sales["family"],
        top_family_sales["total_sales"],
    )
    plt.title("Top 15 Product Families by Total Sales")
    plt.xlabel("Total Sales")
    plt.ylabel("Product Family")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "top_product_families.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        promotion_sales_summary["promotion_status"],
        promotion_sales_summary["average_sales"],
    )
    plt.title("Average Sales by Promotion Status\nDescriptive Comparison Only")
    plt.xlabel("Promotion Status")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "promotion_sales_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def create_data_profile(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Calculate dataset coverage, integrity, and quality statistics."""
    train_data = datasets["train"]
    test_data = datasets["test"]
    stores_data = datasets["stores"]
    oil_data = datasets["oil"]
    holidays_data = datasets["holidays"]
    transactions_data = datasets["transactions"]

    train_start_date = train_data["date"].min()
    train_end_date = train_data["date"].max()
    test_start_date = test_data["date"].min()
    test_end_date = test_data["date"].max()

    train_row_count = len(train_data)
    zero_sales_count = int(train_data["sales"].eq(0).sum())
    negative_sales_count = int(train_data["sales"].lt(0).sum())
    promotion_record_count = int(train_data["onpromotion"].gt(0).sum())

    duplicate_id_count = int(train_data["id"].duplicated().sum())

    duplicate_business_key_count = int(
        train_data.duplicated(
            subset=["date", "store_nbr", "family"]
        ).sum()
    )

    expected_test_row_count = (
        int(test_data["date"].nunique())
        * int(test_data["store_nbr"].nunique())
        * int(test_data["family"].nunique())
    )

    data_profile = {
        "train_dataset": {
            "row_count": train_row_count,
            "column_count": int(train_data.shape[1]),
            "date_start": train_start_date.date().isoformat(),
            "date_end": train_end_date.date().isoformat(),
            "unique_dates": int(train_data["date"].nunique()),
            "unique_stores": int(train_data["store_nbr"].nunique()),
            "unique_families": int(train_data["family"].nunique()),
            "missing_values": get_missing_value_counts(train_data),
        },
        "test_dataset": {
            "row_count": int(len(test_data)),
            "column_count": int(test_data.shape[1]),
            "date_start": test_start_date.date().isoformat(),
            "date_end": test_end_date.date().isoformat(),
            "forecast_horizon_days": int(test_data["date"].nunique()),
            "unique_stores": int(test_data["store_nbr"].nunique()),
            "unique_families": int(test_data["family"].nunique()),
            "expected_complete_grid_rows": expected_test_row_count,
            "is_complete_store_family_date_grid": (
                len(test_data) == expected_test_row_count
            ),
            "days_after_training_end": int(
                (test_start_date - train_end_date).days
            ),
            "missing_values": get_missing_value_counts(test_data),
        },
        "sales_quality": {
            "zero_sales_count": zero_sales_count,
            "zero_sales_percentage": calculate_percentage(
                zero_sales_count,
                train_row_count,
            ),
            "negative_sales_count": negative_sales_count,
            "minimum_sales": float(train_data["sales"].min()),
            "maximum_sales": float(train_data["sales"].max()),
            "average_sales": float(train_data["sales"].mean()),
            "median_sales": float(train_data["sales"].median()),
        },
        "promotion_quality": {
            "promotion_record_count": promotion_record_count,
            "promotion_record_percentage": calculate_percentage(
                promotion_record_count,
                train_row_count,
            ),
            "negative_onpromotion_count": int(
                train_data["onpromotion"].lt(0).sum()
            ),
            "maximum_onpromotion": int(
                train_data["onpromotion"].max()
            ),
        },
        "integrity_checks": {
            "duplicate_train_id_count": duplicate_id_count,
            "duplicate_train_business_key_count": (
                duplicate_business_key_count
            ),
            "store_metadata_duplicate_count": int(
                stores_data["store_nbr"].duplicated().sum()
            ),
            "test_begins_after_training": bool(
                test_start_date > train_end_date
            ),
        },
        "external_datasets": {
            "stores": {
                "row_count": int(len(stores_data)),
                "unique_stores": int(
                    stores_data["store_nbr"].nunique()
                ),
                "missing_values": get_missing_value_counts(
                    stores_data
                ),
            },
            "oil": {
                "row_count": int(len(oil_data)),
                "date_start": oil_data["date"].min().date().isoformat(),
                "date_end": oil_data["date"].max().date().isoformat(),
                "missing_oil_price_count": int(
                    oil_data["dcoilwtico"].isna().sum()
                ),
                "missing_values": get_missing_value_counts(oil_data),
            },
            "holidays": {
                "row_count": int(len(holidays_data)),
                "unique_dates": int(
                    holidays_data["date"].nunique()
                ),
                "transferred_holiday_count": int(
                    holidays_data["transferred"].eq(True).sum()
                ),
                "missing_values": get_missing_value_counts(
                    holidays_data
                ),
            },
            "transactions": {
                "row_count": int(len(transactions_data)),
                "date_start": (
                    transactions_data["date"].min().date().isoformat()
                ),
                "date_end": (
                    transactions_data["date"].max().date().isoformat()
                ),
                "missing_values": get_missing_value_counts(
                    transactions_data
                ),
            },
        },
    }

    return data_profile


def write_markdown_summary(data_profile: dict[str, object]) -> None:
    """Write a readable data-quality summary for documentation."""
    train_profile = data_profile["train_dataset"]
    test_profile = data_profile["test_dataset"]
    sales_profile = data_profile["sales_quality"]
    promotion_profile = data_profile["promotion_quality"]
    integrity_profile = data_profile["integrity_checks"]
    external_profile = data_profile["external_datasets"]

    summary_lines = [
        "# Data Profile Summary",
        "",
        "## Training Data",
        "",
        f"- Rows: {train_profile['row_count']:,}",
        (
            f"- Date range: {train_profile['date_start']} "
            f"to {train_profile['date_end']}"
        ),
        f"- Unique dates: {train_profile['unique_dates']:,}",
        f"- Stores: {train_profile['unique_stores']}",
        f"- Product families: {train_profile['unique_families']}",
        "",
        "## Test Data",
        "",
        f"- Rows: {test_profile['row_count']:,}",
        (
            f"- Date range: {test_profile['date_start']} "
            f"to {test_profile['date_end']}"
        ),
        (
            "- Forecast horizon: "
            f"{test_profile['forecast_horizon_days']} days"
        ),
        (
            "- Complete store-family-date grid: "
            f"{test_profile['is_complete_store_family_date_grid']}"
        ),
        "",
        "## Sales Quality",
        "",
        (
            f"- Zero-sales records: "
            f"{sales_profile['zero_sales_count']:,} "
            f"({sales_profile['zero_sales_percentage']}%)"
        ),
        (
            f"- Negative-sales records: "
            f"{sales_profile['negative_sales_count']:,}"
        ),
        f"- Average sales: {sales_profile['average_sales']:.4f}",
        f"- Median sales: {sales_profile['median_sales']:.4f}",
        f"- Maximum sales: {sales_profile['maximum_sales']:.4f}",
        "",
        "## Promotion Coverage",
        "",
        (
            f"- Promoted records: "
            f"{promotion_profile['promotion_record_count']:,} "
            f"({promotion_profile['promotion_record_percentage']}%)"
        ),
        (
            "- Negative promotion values: "
            f"{promotion_profile['negative_onpromotion_count']}"
        ),
        "",
        "## Integrity Checks",
        "",
        (
            "- Duplicate train IDs: "
            f"{integrity_profile['duplicate_train_id_count']}"
        ),
        (
            "- Duplicate date-store-family keys: "
            f"{integrity_profile['duplicate_train_business_key_count']}"
        ),
        (
            "- Duplicate store metadata records: "
            f"{integrity_profile['store_metadata_duplicate_count']}"
        ),
        (
            "- Test begins after training: "
            f"{integrity_profile['test_begins_after_training']}"
        ),
        "",
        "## External Data",
        "",
        (
            "- Oil-price missing values: "
            f"{external_profile['oil']['missing_oil_price_count']}"
        ),
        (
            "- Transferred holiday records: "
            f"{external_profile['holidays']['transferred_holiday_count']}"
        ),
        "",
        "## Interpretation Warning",
        "",
        (
            "The promotion comparison is descriptive and must not be "
            "interpreted as a causal estimate of promotion effectiveness."
        ),
    ]

    summary_path = DATA_REPORT_DIRECTORY / "data_profile_summary.md"

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete raw-data profiling workflow."""
    DATA_REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    datasets = load_datasets()

    train_data = datasets["train"]

    print(f"Training rows loaded: {len(train_data):,}")
    print(f"Training columns loaded: {train_data.shape[1]}")

    (
        daily_sales_summary,
        family_sales_summary,
        promotion_sales_summary,
    ) = create_summary_tables(train_data)

    daily_sales_summary.to_csv(
        DATA_REPORT_DIRECTORY / "daily_sales_summary.csv",
        index=False,
    )

    family_sales_summary.to_csv(
        DATA_REPORT_DIRECTORY / "family_sales_summary.csv",
        index=False,
    )

    promotion_sales_summary.to_csv(
        DATA_REPORT_DIRECTORY / "promotion_sales_summary.csv",
        index=False,
    )

    data_profile = create_data_profile(datasets)

    profile_path = DATA_REPORT_DIRECTORY / "data_profile.json"

    profile_path.write_text(
        json.dumps(data_profile, indent=2),
        encoding="utf-8",
    )

    write_markdown_summary(data_profile)

    create_figures(
        daily_sales_summary=daily_sales_summary,
        family_sales_summary=family_sales_summary,
        promotion_sales_summary=promotion_sales_summary,
    )

    print("\n" + "=" * 60)
    print("DATA PROFILE COMPLETE")
    print("=" * 60)
    print(f"Profile: {profile_path}")
    print(
        "Summary: "
        f"{DATA_REPORT_DIRECTORY / 'data_profile_summary.md'}"
    )
    print(f"Figures: {FIGURE_DIRECTORY}")
    print(
        "Forecast horizon: "
        f"{data_profile['test_dataset']['forecast_horizon_days']} days"
    )
    print(
        "Duplicate business keys: "
        f"{data_profile['integrity_checks']['duplicate_train_business_key_count']}"
    )


if __name__ == "__main__":
    main()
