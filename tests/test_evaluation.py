import unittest
from unittest.mock import MagicMock

import numpy as np

from adam.evaluation import evaluate_forecast


class ForecastEvaluationTests(unittest.TestCase):
    def test_rolling_forecasts_never_use_future_data(self):
        forecaster = MagicMock()
        def predict(context, prediction_length):
            result = MagicMock()
            result.median = np.array([[[context[-1] + 1] * prediction_length]])
            result.quantile.side_effect = lambda q: result.median
            return result
        forecaster.forecast.side_effect = predict
        result = evaluate_forecast([10, 11, 12, 13, 14], forecaster, initial_window=2, horizon=1)
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['model_mae'], 0)
        self.assertEqual(result['naive_mae'], 1)
        self.assertEqual([len(c.args[0]) for c in forecaster.forecast.call_args_list], [2, 3, 4])
        self.assertEqual(result['predictions'][0]['actual'], 12)

    def test_multistep_and_invalid_parameters(self):
        forecaster = MagicMock()
        output = forecaster.forecast.return_value
        output.median = np.array([[[12., 13.]]])
        output.quantile.return_value = output.median
        result = evaluate_forecast([10, 11, 12, 13], forecaster, initial_window=2, horizon=2)
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['model_mae'], 0)
        self.assertEqual(result['naive_mae'], 2)
        for kwargs in ({'initial_window': 1, 'horizon': 1}, {'initial_window': 4, 'horizon': 1}, {'initial_window': 2, 'horizon': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_forecast([10, 11, 12, 13], forecaster, **kwargs)
