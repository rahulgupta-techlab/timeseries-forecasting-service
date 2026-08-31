from __future__ import annotations

import numpy as np
import pytest

from src.forecasting import fit_and_forecast


def test_forecast_returns_correct_number_of_points():
    values = list(range(1, 21))  # clean linear trend: 1..20
    horizon = 5
    forecast = fit_and_forecast(values, horizon)
    assert len(forecast) == horizon


def test_forecast_continues_linear_trend_reasonably():
    # A clean linear trend with slope 2, no noise.
    values = [2.0 * i for i in range(1, 31)]  # 2, 4, ..., 60
    horizon = 3
    forecast = fit_and_forecast(values, horizon)

    expected = [62.0, 64.0, 66.0]
    for got, exp in zip(forecast, expected):
        assert got == pytest.approx(exp, abs=1.5)


def test_forecast_raises_on_invalid_horizon():
    with pytest.raises(ValueError):
        fit_and_forecast([1.0, 2.0, 3.0], 0)


def test_forecast_raises_on_too_short_series():
    with pytest.raises(ValueError):
        fit_and_forecast([1.0], 3)


def test_forecast_handles_short_series_without_trend_component():
    # Only 3 points: falls back to simple exponential smoothing internally.
    forecast = fit_and_forecast([10.0, 12.0, 11.0], 2)
    assert len(forecast) == 2
    assert all(np.isfinite(v) for v in forecast)
