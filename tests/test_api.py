from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_endpoint_with_sample_series():
    response = client.post("/forecast", json={"horizon": 5})
    assert response.status_code == 200

    body = response.json()
    assert body["horizon"] == 5
    assert len(body["forecast"]) == 5
    assert len(body["dates"]) == 5
    assert all(isinstance(v, (int, float)) for v in body["forecast"])


def test_forecast_endpoint_with_custom_series():
    custom_series = [float(i) for i in range(1, 21)]
    response = client.post("/forecast", json={"horizon": 3, "series": custom_series})
    assert response.status_code == 200

    body = response.json()
    assert body["horizon"] == 3
    assert len(body["forecast"]) == 3


def test_forecast_endpoint_rejects_invalid_horizon():
    response = client.post("/forecast", json={"horizon": 0})
    assert response.status_code == 422


def test_forecast_endpoint_rejects_too_short_custom_series():
    response = client.post("/forecast", json={"horizon": 2, "series": [1.0]})
    assert response.status_code == 400
