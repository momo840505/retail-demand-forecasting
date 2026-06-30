from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def get_valid_selection() -> tuple[int, str]:
    stores_response = client.get("/stores")
    families_response = client.get("/families")

    assert stores_response.status_code == 200
    assert families_response.status_code == 200

    stores = stores_response.json()
    families = families_response.json()

    assert stores
    assert families

    return int(stores[0]["store_nbr"]), str(families[0])


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"
    assert result["forecast_rows"] > 0


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    result = response.json()

    assert result["model"] == "xgboost_log_target_nested"
    assert result["forecast_horizon_days"] == 16
    assert result["fold_count"] == 4
    assert result["pooled_wape_percentage"] > 0
    assert result["pooled_rmsle"] > 0


def test_monitoring_info_endpoint() -> None:
    response = client.get("/monitoring-info")

    assert response.status_code == 200

    result = response.json()

    assert result["model_version"] == "xgboost_log_target_nested"
    assert "forecast_row_count" in result["monitored_signals"]
    assert result["data_quality_checks"]
    assert result["model_quality_checks"]
    assert result["operational_checks"]
    assert result["retraining_triggers"]


def test_forecast_endpoint() -> None:
    store_nbr, family = get_valid_selection()

    response = client.get(
        "/forecasts",
        params={
            "store_nbr": store_nbr,
            "family": family,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["store_nbr"] == store_nbr
    assert result["family"] == family
    assert result["forecast_horizon_days"] == 16
    assert len(result["forecasts"]) == 16
    assert result["total_predicted_sales"] >= 0
    assert result["average_daily_demand"] >= 0


def test_replenishment_endpoint() -> None:
    store_nbr, family = get_valid_selection()

    response = client.post(
        "/replenishment",
        json={
            "store_nbr": store_nbr,
            "family": family,
            "current_inventory": 20,
            "inbound_inventory": 0,
            "lead_time_days": 3,
            "safety_stock_days": 2,
            "review_period_days": 7,
            "case_pack_size": 6,
            "minimum_order_quantity": 12,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["store_nbr"] == store_nbr
    assert result["family"] == family
    assert result["forecast_coverage_days"] == 16
    assert result["suggested_order_quantity"] >= 0

    assert result["stockout_risk_band"] in {
        "Critical",
        "High",
        "Moderate",
        "Low",
    }


def test_unknown_store_returns_not_found() -> None:
    response = client.get(
        "/forecasts",
        params={
            "store_nbr": 9999,
            "family": "AUTOMOTIVE",
        },
    )

    assert response.status_code == 404
