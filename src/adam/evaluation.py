import numpy as np

from adam.forecast import forecast_evidence


def evaluate_forecast(prices, forecaster, *, initial_window: int, horizon: int = 1, step: int = 1) -> dict:
    values = np.asarray(prices, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any()
            or type(initial_window) is not int or initial_window < 2
            or type(horizon) is not int or horizon < 1
            or type(step) is not int or step < 1 or initial_window + horizon > len(values)):
        raise ValueError("Invalid price history or evaluation windows")
    predictions = []
    for end in range(initial_window, len(values) - horizon + 1, step):
        context = values[:end]
        forecast = forecast_evidence(context, horizon, forecaster)
        predicted = float(context[-1] * (1 + forecast['median_returns'][-1]))
        predictions.append({'origin_index': end - 1, 'target_index': end + horizon - 1,
                            'actual': float(values[end + horizon - 1]),
                            'model': predicted, 'naive': float(context[-1])})
    model_errors = np.array([row['model'] - row['actual'] for row in predictions])
    naive_errors = np.array([row['naive'] - row['actual'] for row in predictions])
    return {'horizon': horizon, 'samples': len(predictions), 'predictions': predictions,
            'model_mae': float(np.mean(np.abs(model_errors))),
            'naive_mae': float(np.mean(np.abs(naive_errors))),
            'model_rmse': float(np.sqrt(np.mean(model_errors ** 2))),
            'naive_rmse': float(np.sqrt(np.mean(naive_errors ** 2)))}
