from __future__ import annotations

import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
DASHBOARD_DATA_DIRECTORY = PROJECT_ROOT / "dashboard" / "data"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))


from retail_forecasting.replenishment import (  # noqa: E402
    ReplenishmentInputs,
    calculate_replenishment_plan,
)


# =============================================================================
# API response models
# =============================================================================


class HealthResponse(BaseModel):
    status: str
    service: str
    forecast_rows: int


class StoreResponse(BaseModel):
    store_nbr: int
    city: str
    state: str
    type: str
    cluster: int


class ModelInfoResponse(BaseModel):
    model: str
    validation_strategy: str
    training_window_days: int
    inner_validation_days: int
    forecast_horizon_days: int
    fold_count: int
    pooled_wape_percentage: float
    pooled_rmsle: float
    wape_improvement_vs_best_baseline_percentage: float
    rmsle_improvement_vs_best_baseline_percentage: float
    forecast_date_start: date
    forecast_date_end: date
    test_labels_available: bool


class ForecastPoint(BaseModel):
    date: date
    predicted_sales: float
    onpromotion: float


class ForecastResponse(BaseModel):
    store_nbr: int
    family: str
    city: str
    state: str
    forecast_horizon_days: int
    total_predicted_sales: float
    average_daily_demand: float
    forecasts: list[ForecastPoint]


class ReplenishmentRequest(BaseModel):
    store_nbr: int = Field(ge=1)
    family: str = Field(min_length=1)
    current_inventory: float = Field(ge=0)
    inbound_inventory: float = Field(default=0.0, ge=0)
    lead_time_days: int = Field(default=3, ge=1, le=16)
    safety_stock_days: int = Field(default=2, ge=0, le=16)
    review_period_days: int = Field(default=7, ge=1, le=16)
    case_pack_size: int = Field(default=1, ge=1)
    minimum_order_quantity: int = Field(default=0, ge=0)


class ReplenishmentResponse(BaseModel):
    store_nbr: int
    family: str
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


# =============================================================================
# FastAPI application
# =============================================================================


app = FastAPI(
    title="Retail Demand Forecasting API",
    version="1.0.0",
    description=(
        "Read-only retail demand forecasts and rule-based replenishment "
        "decision support for the portfolio demo."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# Data loading
# =============================================================================


@lru_cache(maxsize=1)
def load_api_data() -> dict[str, Any]:
    """Load and cache the deployment-ready forecast data."""

    paths = {
        "forecast": DASHBOARD_DATA_DIRECTORY / "forecast.parquet",
        "stores": DASHBOARD_DATA_DIRECTORY / "stores.csv",
        "summary": DASHBOARD_DATA_DIRECTORY / "model_summary.csv",
        "manifest": DASHBOARD_DATA_DIRECTORY / "dashboard_manifest.json",
    }

    missing_files = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "API data is incomplete. Missing files:\n"
            + "\n".join(missing_files)
        )

    forecast_data = pd.read_parquet(paths["forecast"])

    required_forecast_columns = {
        "date",
        "store_nbr",
        "family",
        "predicted_sales",
        "onpromotion",
    }

    missing_forecast_columns = (
        required_forecast_columns - set(forecast_data.columns)
    )

    if missing_forecast_columns:
        raise ValueError(
            "Forecast data is missing columns: "
            f"{sorted(missing_forecast_columns)}"
        )

    forecast_data = forecast_data.copy()

    forecast_data["date"] = pd.to_datetime(
        forecast_data["date"]
    )

    forecast_data["store_nbr"] = pd.to_numeric(
        forecast_data["store_nbr"],
        errors="raise",
    ).astype(int)

    forecast_data["family"] = (
        forecast_data["family"].astype(str)
    )

    forecast_data["predicted_sales"] = pd.to_numeric(
        forecast_data["predicted_sales"],
        errors="raise",
    )

    forecast_data["onpromotion"] = pd.to_numeric(
        forecast_data["onpromotion"],
        errors="coerce",
    ).fillna(0.0)

    store_data = pd.read_csv(paths["stores"])

    required_store_columns = {
        "store_nbr",
        "city",
        "state",
        "type",
        "cluster",
    }

    missing_store_columns = (
        required_store_columns - set(store_data.columns)
    )

    if missing_store_columns:
        raise ValueError(
            "Store data is missing columns: "
            f"{sorted(missing_store_columns)}"
        )

    store_data["store_nbr"] = pd.to_numeric(
        store_data["store_nbr"],
        errors="raise",
    ).astype(int)

    model_summary_data = pd.read_csv(paths["summary"])

    if model_summary_data.empty:
        raise ValueError("Model summary cannot be empty.")

    manifest = json.loads(
        paths["manifest"].read_text(encoding="utf-8")
    )

    return {
        "forecast": forecast_data,
        "stores": store_data,
        "summary": model_summary_data.iloc[0],
        "manifest": manifest,
    }


def get_api_data() -> dict[str, Any]:
    """Return cached data or convert loading errors into HTTP errors."""

    try:
        return load_api_data()

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


def get_forecast_subset(
    store_nbr: int,
    family: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return one store-family forecast series."""

    api_data = get_api_data()

    forecast_data = api_data["forecast"]
    store_data = api_data["stores"]

    store_match = store_data.loc[
        store_data["store_nbr"].eq(store_nbr)
    ]

    if store_match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Store {store_nbr} was not found.",
        )

    normalized_family = family.strip().casefold()

    forecast_subset = forecast_data.loc[
        forecast_data["store_nbr"].eq(store_nbr)
        & forecast_data["family"]
        .str.casefold()
        .eq(normalized_family)
    ].sort_values("date")

    if forecast_subset.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast was found for store {store_nbr} "
                f"and family '{family}'."
            ),
        )

    return forecast_subset, store_match.iloc[0]


# =============================================================================
# Service endpoints
# =============================================================================


@app.get(
    "/",
    tags=["Service"],
)
def read_root() -> dict[str, str]:
    return {
        "service": "Retail Demand Forecasting API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
)
def health_check() -> HealthResponse:
    api_data = get_api_data()

    return HealthResponse(
        status="ok",
        service="retail-demand-forecasting-api",
        forecast_rows=len(api_data["forecast"]),
    )


# =============================================================================
# Model endpoint
# =============================================================================


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def get_model_info() -> ModelInfoResponse:
    api_data = get_api_data()

    model_summary = api_data["summary"]
    manifest = api_data["manifest"]

    return ModelInfoResponse(
        model=str(model_summary["model"]),
        validation_strategy=str(
            model_summary["validation_strategy"]
        ),
        training_window_days=int(
            model_summary["training_window_days"]
        ),
        inner_validation_days=int(
            model_summary["inner_validation_days"]
        ),
        forecast_horizon_days=int(
            model_summary["forecast_horizon_days"]
        ),
        fold_count=int(
            model_summary["fold_count"]
        ),
        pooled_wape_percentage=float(
            model_summary["pooled_wape_percentage"]
        ),
        pooled_rmsle=float(
            model_summary["pooled_rmsle"]
        ),
        wape_improvement_vs_best_baseline_percentage=float(
            model_summary[
                "wape_improvement_vs_best_baseline_percentage"
            ]
        ),
        rmsle_improvement_vs_best_baseline_percentage=float(
            model_summary[
                "rmsle_improvement_vs_best_baseline_percentage"
            ]
        ),
        forecast_date_start=pd.Timestamp(
            manifest["forecast_date_start"]
        ).date(),
        forecast_date_end=pd.Timestamp(
            manifest["forecast_date_end"]
        ).date(),
        test_labels_available=bool(
            manifest["test_labels_available"]
        ),
    )


# =============================================================================
# Reference endpoints
# =============================================================================


@app.get(
    "/stores",
    response_model=list[StoreResponse],
    tags=["Reference"],
)
def list_stores() -> list[StoreResponse]:
    store_data = (
        get_api_data()["stores"]
        .sort_values("store_nbr")
    )

    return [
        StoreResponse(
            store_nbr=int(row.store_nbr),
            city=str(row.city),
            state=str(row.state),
            type=str(row.type),
            cluster=int(row.cluster),
        )
        for row in store_data.itertuples(index=False)
    ]


@app.get(
    "/families",
    response_model=list[str],
    tags=["Reference"],
)
def list_families() -> list[str]:
    forecast_data = get_api_data()["forecast"]

    return sorted(
        forecast_data["family"]
        .astype(str)
        .unique()
        .tolist()
    )


# =============================================================================
# Forecast endpoint
# =============================================================================


@app.get(
    "/forecasts",
    response_model=ForecastResponse,
    tags=["Forecasts"],
)
def get_forecast(
    store_nbr: int = Query(ge=1),
    family: str = Query(min_length=1),
) -> ForecastResponse:
    forecast_subset, store_row = get_forecast_subset(
        store_nbr=store_nbr,
        family=family,
    )

    canonical_family = str(
        forecast_subset["family"].iloc[0]
    )

    forecast_points = [
        ForecastPoint(
            date=pd.Timestamp(row.date).date(),
            predicted_sales=float(row.predicted_sales),
            onpromotion=float(row.onpromotion),
        )
        for row in forecast_subset.itertuples(index=False)
    ]

    return ForecastResponse(
        store_nbr=store_nbr,
        family=canonical_family,
        city=str(store_row["city"]),
        state=str(store_row["state"]),
        forecast_horizon_days=len(forecast_points),
        total_predicted_sales=float(
            forecast_subset["predicted_sales"].sum()
        ),
        average_daily_demand=float(
            forecast_subset["predicted_sales"].mean()
        ),
        forecasts=forecast_points,
    )


# =============================================================================
# Replenishment endpoint
# =============================================================================


@app.post(
    "/replenishment",
    response_model=ReplenishmentResponse,
    tags=["Replenishment"],
)
def create_replenishment_plan(
    request: ReplenishmentRequest,
) -> ReplenishmentResponse:
    forecast_subset, _ = get_forecast_subset(
        store_nbr=request.store_nbr,
        family=request.family,
    )

    replenishment_inputs = ReplenishmentInputs(
        current_inventory=request.current_inventory,
        inbound_inventory=request.inbound_inventory,
        lead_time_days=request.lead_time_days,
        safety_stock_days=request.safety_stock_days,
        review_period_days=request.review_period_days,
        case_pack_size=request.case_pack_size,
        minimum_order_quantity=(
            request.minimum_order_quantity
        ),
    )

    try:
        plan = calculate_replenishment_plan(
            forecast_data=forecast_subset[
                ["date", "predicted_sales"]
            ],
            inputs=replenishment_inputs,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    canonical_family = str(
        forecast_subset["family"].iloc[0]
    )

    return ReplenishmentResponse(
        store_nbr=request.store_nbr,
        family=canonical_family,
        **plan.to_dict(),
    )