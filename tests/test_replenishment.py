import pandas as pd
import pytest

from retail_forecasting.replenishment import (
    ReplenishmentInputs,
    calculate_replenishment_plan,
)


def create_constant_forecast(
    daily_demand: float,
    day_count: int = 16,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=day_count,
            ),
            "predicted_sales": [
                daily_demand
            ] * day_count,
        }
    )


def test_critical_risk_generates_rounded_order() -> None:
    forecast_data = create_constant_forecast(
        daily_demand=10.0
    )

    inputs = ReplenishmentInputs(
        current_inventory=15.0,
        inbound_inventory=0.0,
        lead_time_days=3,
        safety_stock_days=2,
        review_period_days=4,
        case_pack_size=6,
        minimum_order_quantity=12,
    )

    plan = calculate_replenishment_plan(
        forecast_data=forecast_data,
        inputs=inputs,
    )

    assert plan.average_daily_demand == 10.0
    assert plan.lead_time_demand == 30.0
    assert plan.safety_stock_units == 20.0
    assert plan.reorder_point == 50.0
    assert plan.target_inventory_level == 90.0
    assert plan.raw_order_quantity == 75.0
    assert plan.suggested_order_quantity == 78
    assert plan.stockout_risk_band == "Critical"
    assert plan.reorder_now is True


def test_sufficient_inventory_requires_no_order() -> None:
    forecast_data = create_constant_forecast(
        daily_demand=10.0
    )

    inputs = ReplenishmentInputs(
        current_inventory=100.0,
        lead_time_days=2,
        safety_stock_days=1,
        review_period_days=3,
        case_pack_size=5,
    )

    plan = calculate_replenishment_plan(
        forecast_data=forecast_data,
        inputs=inputs,
    )

    assert plan.reorder_point == 30.0
    assert plan.target_inventory_level == 60.0
    assert plan.suggested_order_quantity == 0
    assert plan.stockout_risk_band == "Low"
    assert plan.reorder_now is False
    assert plan.days_of_cover == 10.0


def test_forecast_must_cover_protection_period() -> None:
    forecast_data = create_constant_forecast(
        daily_demand=10.0,
        day_count=5,
    )

    inputs = ReplenishmentInputs(
        current_inventory=20.0,
        lead_time_days=3,
        review_period_days=4,
    )

    with pytest.raises(ValueError):
        calculate_replenishment_plan(
            forecast_data=forecast_data,
            inputs=inputs,
        )


def test_negative_inventory_is_rejected() -> None:
    forecast_data = create_constant_forecast(
        daily_demand=10.0
    )

    inputs = ReplenishmentInputs(
        current_inventory=-1.0,
    )

    with pytest.raises(ValueError):
        calculate_replenishment_plan(
            forecast_data=forecast_data,
            inputs=inputs,
        )
