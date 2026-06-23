from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ReplenishmentInputs:
    """User-provided operational assumptions."""

    current_inventory: float
    inbound_inventory: float = 0.0
    lead_time_days: int = 3
    safety_stock_days: int = 2
    review_period_days: int = 7
    case_pack_size: int = 1
    minimum_order_quantity: int = 0


@dataclass(frozen=True)
class ReplenishmentPlan:
    """Deterministic replenishment decision-support output."""

    forecast_coverage_days: int
    average_daily_demand: float
    inventory_position: float
    lead_time_demand: float
    safety_stock_units: float
    reorder_point: float
    protection_period_demand: float
    target_inventory_level: float
    projected_inventory_at_arrival: float
    days_of_cover: float | None
    reorder_now: bool
    raw_order_quantity: float
    suggested_order_quantity: int
    stockout_risk_band: str

    def to_dict(self) -> dict[str, object]:
        """Convert the plan into a serialisable dictionary."""
        return asdict(self)


def validate_replenishment_inputs(
    inputs: ReplenishmentInputs,
) -> None:
    """Validate operational assumptions."""
    if inputs.current_inventory < 0:
        raise ValueError(
            "Current inventory cannot be negative."
        )

    if inputs.inbound_inventory < 0:
        raise ValueError(
            "Inbound inventory cannot be negative."
        )

    if inputs.lead_time_days <= 0:
        raise ValueError(
            "Lead time must be greater than zero."
        )

    if inputs.safety_stock_days < 0:
        raise ValueError(
            "Safety-stock days cannot be negative."
        )

    if inputs.review_period_days <= 0:
        raise ValueError(
            "Review period must be greater than zero."
        )

    if inputs.case_pack_size <= 0:
        raise ValueError(
            "Case-pack size must be greater than zero."
        )

    if inputs.minimum_order_quantity < 0:
        raise ValueError(
            "Minimum order quantity cannot be negative."
        )


def prepare_forecast_data(
    forecast_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and sort one store-family forecast series."""
    required_columns = {
        "date",
        "predicted_sales",
    }

    missing_columns = (
        required_columns
        - set(forecast_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Forecast data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    result = forecast_data[
        [
            "date",
            "predicted_sales",
        ]
    ].copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result["predicted_sales"] = pd.to_numeric(
        result["predicted_sales"],
        errors="raise",
    )

    if result.empty:
        raise ValueError(
            "Forecast data cannot be empty."
        )

    if result["date"].duplicated().any():
        raise ValueError(
            "Forecast dates must be unique."
        )

    if result["predicted_sales"].isna().any():
        raise ValueError(
            "Forecast values cannot be missing."
        )

    if (
        result["predicted_sales"] < 0
    ).any():
        raise ValueError(
            "Forecast values cannot be negative."
        )

    return result.sort_values(
        "date"
    ).reset_index(drop=True)


def round_order_quantity(
    raw_order_quantity: float,
    case_pack_size: int,
    minimum_order_quantity: int,
) -> int:
    """Apply minimum order and case-pack constraints."""
    if raw_order_quantity <= 0:
        return 0

    constrained_quantity = max(
        raw_order_quantity,
        float(minimum_order_quantity),
    )

    return int(
        math.ceil(
            constrained_quantity
            / case_pack_size
        )
        * case_pack_size
    )


def determine_stockout_risk_band(
    inventory_position: float,
    lead_time_demand: float,
    reorder_point: float,
    target_inventory_level: float,
) -> str:
    """
    Assign a deterministic risk band.

    This is a rule-based classification, not a probability estimate.
    """
    if inventory_position < lead_time_demand:
        return "Critical"

    if inventory_position <= reorder_point:
        return "High"

    if inventory_position < target_inventory_level:
        return "Moderate"

    return "Low"


def calculate_replenishment_plan(
    forecast_data: pd.DataFrame,
    inputs: ReplenishmentInputs,
) -> ReplenishmentPlan:
    """Calculate a deterministic replenishment recommendation."""
    validate_replenishment_inputs(inputs)

    prepared_forecast = prepare_forecast_data(
        forecast_data
    )

    protection_period_days = (
        inputs.lead_time_days
        + inputs.review_period_days
    )

    if (
        protection_period_days
        > len(prepared_forecast)
    ):
        raise ValueError(
            "The forecast does not cover the complete "
            "lead-time and review period."
        )

    average_daily_demand = float(
        prepared_forecast[
            "predicted_sales"
        ].mean()
    )

    lead_time_demand = float(
        prepared_forecast[
            "predicted_sales"
        ]
        .iloc[:inputs.lead_time_days]
        .sum()
    )

    protection_period_demand = float(
        prepared_forecast[
            "predicted_sales"
        ]
        .iloc[:protection_period_days]
        .sum()
    )

    safety_stock_units = (
        average_daily_demand
        * inputs.safety_stock_days
    )

    inventory_position = (
        inputs.current_inventory
        + inputs.inbound_inventory
    )

    reorder_point = (
        lead_time_demand
        + safety_stock_units
    )

    target_inventory_level = (
        protection_period_demand
        + safety_stock_units
    )

    projected_inventory_at_arrival = (
        inventory_position
        - lead_time_demand
    )

    if average_daily_demand > 0:
        days_of_cover: float | None = (
            inventory_position
            / average_daily_demand
        )
    else:
        days_of_cover = None

    reorder_now = (
        inventory_position <= reorder_point
    )

    if reorder_now:
        raw_order_quantity = max(
            0.0,
            target_inventory_level
            - inventory_position,
        )
    else:
        raw_order_quantity = 0.0

    suggested_order_quantity = (
        round_order_quantity(
            raw_order_quantity=(
                raw_order_quantity
            ),
            case_pack_size=(
                inputs.case_pack_size
            ),
            minimum_order_quantity=(
                inputs.minimum_order_quantity
            ),
        )
    )

    stockout_risk_band = (
        determine_stockout_risk_band(
            inventory_position=(
                inventory_position
            ),
            lead_time_demand=(
                lead_time_demand
            ),
            reorder_point=(
                reorder_point
            ),
            target_inventory_level=(
                target_inventory_level
            ),
        )
    )

    return ReplenishmentPlan(
        forecast_coverage_days=int(
            len(prepared_forecast)
        ),
        average_daily_demand=(
            average_daily_demand
        ),
        inventory_position=(
            inventory_position
        ),
        lead_time_demand=(
            lead_time_demand
        ),
        safety_stock_units=(
            safety_stock_units
        ),
        reorder_point=reorder_point,
        protection_period_demand=(
            protection_period_demand
        ),
        target_inventory_level=(
            target_inventory_level
        ),
        projected_inventory_at_arrival=(
            projected_inventory_at_arrival
        ),
        days_of_cover=days_of_cover,
        reorder_now=reorder_now,
        raw_order_quantity=(
            raw_order_quantity
        ),
        suggested_order_quantity=(
            suggested_order_quantity
        ),
        stockout_risk_band=(
            stockout_risk_band
        ),
    )
