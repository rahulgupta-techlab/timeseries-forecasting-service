"""Backtesting utility: hold out the last N points, forecast them, score the result."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.forecasting import fit_and_forecast


def backtest(values: Sequence[float], holdout: int) -> dict:
    """Hold out the last `holdout` points of `values`, fit on the rest, forecast
    `holdout` steps, and compare against the actual held-out values.

    Args:
        values: the full historical series, ordered oldest to newest.
        holdout: number of trailing points to hold out and evaluate against.
            Must be >= 1 and leave at least 2 points for training.

    Returns:
        A dict with keys:
            "mae": mean absolute error between forecast and actual holdout.
            "rmse": root mean squared error between forecast and actual holdout.
            "forecast": the list of forecasted values for the holdout period.
            "actual": the list of actual held-out values.
    """
    series = list(values)
    if holdout < 1:
        raise ValueError("holdout must be >= 1")
    if len(series) - holdout < 2:
        raise ValueError("not enough data to train after holding out the requested points")

    train = series[:-holdout]
    actual = series[-holdout:]

    forecast = fit_and_forecast(train, holdout)

    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)

    errors = forecast_arr - actual_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "forecast": forecast,
        "actual": actual,
    }
