"""Core forecasting logic built on statsmodels' Holt-Winters exponential smoothing.

The sample series in this project is short (a few months of daily data) and
does not carry a strong, stable weekly seasonal cycle, so a full seasonal
Holt-Winters fit is prone to convergence warnings on such data. To keep the
model robust and warning-free across arbitrary input series, this module uses
Holt's linear trend method (exponential smoothing with a trend component but
no seasonal component). It is still `statsmodels.tsa.holtwinters.ExponentialSmoothing`
under the hood, just configured without the seasonal term.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_and_forecast(values: Sequence[float], horizon: int) -> list[float]:
    """Fit an exponential smoothing model on `values` and forecast `horizon` steps ahead.

    Uses Holt's linear trend method (additive trend, no seasonality), which is
    robust on short series and doesn't require a seasonal period to be
    specified. Falls back to simple exponential smoothing (no trend) if there
    isn't enough data to fit a trend component.

    Args:
        values: historical observations, ordered oldest to newest.
        horizon: number of future periods to forecast. Must be >= 1.

    Returns:
        A list of `horizon` forecasted values.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    series = np.asarray(values, dtype=float)
    if series.size < 2:
        raise ValueError("need at least 2 observations to fit a model")

    if series.size >= 4:
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
    else:
        # Too short for a reliable trend estimate; fall back to simple
        # exponential smoothing (level only).
        model = ExponentialSmoothing(series, trend=None, seasonal=None, initialization_method="estimated")

    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(horizon)
    return [float(v) for v in forecast]
