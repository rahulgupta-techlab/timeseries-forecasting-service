# Time Series Forecasting Service

A small, self-contained time-series forecasting API: fit an exponential smoothing
model on a historical series and forecast the next N periods. Includes a
backtesting utility for evaluating forecast accuracy (MAE/RMSE) against known
held-out data, and a FastAPI endpoint for serving forecasts on demand.

The core model is `statsmodels`' `ExponentialSmoothing`, configured as Holt's
linear trend method (level + trend, no seasonal component). The bundled sample
series is short (90 days) and its weekly seasonal signal is not strong or
stable enough for a full seasonal Holt-Winters fit to converge reliably, so
seasonality is intentionally left out in favor of a model that fits cleanly
and produces sane forecasts on any short series passed in. For very short
series (fewer than 4 points) the model automatically falls back to simple
exponential smoothing (level only, no trend).

## Stack

- Python 3.11
- `statsmodels` for exponential smoothing (Holt-Winters family)
- `pandas` / `numpy` for data handling
- FastAPI for the forecast endpoint

## Project layout

```
src/
  forecasting.py   # core model fit/predict logic (Holt's linear trend)
  backtest.py       # backtesting utility: holdout, forecast, MAE/RMSE
  main.py            # FastAPI app exposing POST /forecast
data/
  sample_timeseries.csv   # synthetic daily series: trend + weekly seasonality + noise
tests/                     # unit tests for forecasting, backtest, and the API
```

## Running it

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload        # serves POST /forecast on http://localhost:8000
```

```bash
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" \
  -d '{"horizon": 7}'
```

This forecasts 7 days ahead using the bundled sample series. To forecast from
your own data, pass a `series` array instead:

```bash
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" \
  -d '{"horizon": 5, "series": [10.0, 12.0, 11.5, 13.0, 14.2, 13.8, 15.1]}'
```

## Tests

```bash
pytest
```

## Notes

This is a portfolio/demo project - the forecasting model is intentionally kept
simple (a single well-understood exponential smoothing method) so the overall
pipeline (data -> model -> backtest -> API) is easy to follow and extend, e.g.
by swapping in a different `statsmodels` or `prophet`-style model behind the
same `fit_and_forecast` interface.

## License

Copyright (c) 2026 Rahul Gupta. All rights reserved. Published for viewing and evaluation only; copying or reuse requires written permission. See [LICENSE](LICENSE).
