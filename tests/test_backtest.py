from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import backtest


def test_backtest_computes_mae_and_rmse_correctly(monkeypatch):
    # Use a hand-computable example: fix the forecast returned by
    # fit_and_forecast so we can verify the MAE/RMSE math independently of
    # the underlying model.
    import src.backtest as backtest_module

    def fake_forecast(values, horizon):
        return [10.0, 20.0, 30.0]

    monkeypatch.setattr(backtest_module, "fit_and_forecast", fake_forecast)

    series = [1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 18.0, 33.0]
    result = backtest(series, holdout=3)

    # actual held-out values = [12.0, 18.0, 33.0]
    # forecast = [10.0, 20.0, 30.0]
    # errors = [-2.0, 2.0, -3.0]
    # MAE = (2 + 2 + 3) / 3 = 2.3333...
    # RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(17/3) = 2.3804...
    assert result["mae"] == pytest.approx(7.0 / 3.0, rel=1e-9)
    assert result["rmse"] == pytest.approx((17.0 / 3.0) ** 0.5, rel=1e-9)
    assert result["forecast"] == [10.0, 20.0, 30.0]
    assert result["actual"] == [12.0, 18.0, 33.0]


def test_backtest_runs_without_error_on_sample_dataset():
    df = pd.read_csv("data/sample_timeseries.csv")
    values = df["value"].tolist()

    result = backtest(values, holdout=7)

    assert set(result.keys()) == {"mae", "rmse", "forecast", "actual"}
    assert len(result["forecast"]) == 7
    assert len(result["actual"]) == 7
    assert result["mae"] >= 0
    assert result["rmse"] >= 0


def test_backtest_raises_on_insufficient_data():
    with pytest.raises(ValueError):
        backtest([1.0, 2.0, 3.0], holdout=2)
