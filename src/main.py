"""FastAPI app exposing POST /forecast for the time-series forecasting service."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.forecasting import fit_and_forecast

app = FastAPI(title="Time Series Forecasting Service")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_timeseries.csv"


def load_sample_series() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


class ForecastRequest(BaseModel):
    horizon: int = Field(..., ge=1, le=365, description="Number of future periods to forecast")
    series: Optional[list[float]] = Field(
        default=None,
        description="Optional custom series of historical values. Defaults to the bundled sample series.",
    )


class ForecastResponse(BaseModel):
    horizon: int
    forecast: list[float]
    dates: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
    if request.series is not None:
        values = request.series
        if len(values) < 2:
            raise HTTPException(status_code=400, detail="series must contain at least 2 values")
        last_date = datetime.now(timezone.utc).date()
    else:
        df = load_sample_series()
        values = df["value"].tolist()
        last_date = df["date"].iloc[-1].date()

    try:
        forecast_values = fit_and_forecast(values, request.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    future_dates = [
        (last_date + timedelta(days=i + 1)).isoformat() for i in range(request.horizon)
    ]

    return ForecastResponse(horizon=request.horizon, forecast=forecast_values, dates=future_dates)
