import threading

import numpy as np

_forecast_lock = threading.Lock()
_model_lock = threading.Lock()
_forecaster = None


def load_forecaster(model: str = "mistralai/Timestral-5M", *, device: str = "cpu", local_files_only: bool = False):
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(model, device=device, local_files_only=local_files_only)


def shared_forecaster():
    global _forecaster
    with _model_lock:
        if _forecaster is None:
            _forecaster = load_forecaster()
        return _forecaster


def forecast_evidence(prices, horizon: int, forecaster) -> dict:
    values = np.array(prices, dtype=float, copy=True)
    if (values.ndim != 1 or values.size < 2 or not np.isfinite(values).all()
            or (values <= 0).any() or type(horizon) is not int or horizon < 1):
        raise ValueError("A positive price series and forecast horizon are required")
    with _forecast_lock:
        forecast = forecaster.forecast(values, prediction_length=horizon)
    arrays = [np.asarray(x, dtype=float).reshape(-1) for x in
              (forecast.quantile(0.1), forecast.median, forecast.quantile(0.9))]
    if any(a.size != horizon or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid Timestral forecast")
    lower, median, upper = arrays
    if np.any(lower > median) or np.any(median > upper):
        raise ValueError("Timestral quantiles are out of order")
    return {
        "last_close": float(values[-1]),
        "horizons": list(range(1, horizon + 1)),
        "lower_returns": (lower / values[-1] - 1).tolist(),
        "median_returns": (median / values[-1] - 1).tolist(),
        "upper_returns": (upper / values[-1] - 1).tolist(),
        "quantile_levels": [0.1, 0.5, 0.9],
    }


async def assess_forecast(ticker, evidence, model):
    return await model.assess(ticker, "forecast", evidence)
