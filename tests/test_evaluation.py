import unittest
from unittest.mock import MagicMock

import numpy as np

from adam.evaluation import evaluate_forecast


class ForecastEvaluationTests(unittest.TestCase):
    def test_direction_and_quantile_coverage_are_measured_at_each_origin(self):
        forecaster = MagicMock()
        def predict(context, prediction_length):
            origin = context[-1]
            median = {11.: 12. if len(context) == 2 else 11., 12.: 13.}[origin]
            lower = 11. if len(context) == 2 else 10.
            upper = 13. if len(context) == 2 else 14. if len(context) == 3 else 12.
            result = MagicMock()
            result.median = np.array([[[median]]])
            result.quantile.side_effect = lambda q: np.array([[[lower if q == .1 else upper]]])
            return result
        forecaster.forecast.side_effect = predict
        result = evaluate_forecast([10, 11, 12, 11, 11], forecaster, initial_window=2)
        self.assertEqual(result['directional_samples'], 2)
        self.assertEqual(result['directional_accuracy'], .5)
        self.assertEqual(result['interval_coverage'], 1.)
        self.assertEqual(result['lower_quantile_coverage'], 0.)
        self.assertEqual(result['upper_quantile_coverage'], 1.)
        self.assertAlmostEqual(result['mean_interval_width'], 8 / 3)
        self.assertEqual(result['predictions'][0]['lower'], 11.)
        self.assertEqual(result['predictions'][0]['upper'], 13.)

    def test_directional_accuracy_is_undefined_when_prices_do_not_change(self):
        forecaster = MagicMock()
        forecaster.forecast.return_value.median = np.array([[[10.]]])
        forecaster.forecast.return_value.quantile.return_value = np.array([[[10.]]])
        result = evaluate_forecast([10, 10, 10], forecaster, initial_window=2)
        self.assertEqual(result['directional_samples'], 0)
        self.assertIsNone(result['directional_accuracy'])
        self.assertEqual(result['interval_coverage'], 1.)

    def test_interval_miss_and_flat_prediction_are_not_counted_as_success(self):
        forecaster = MagicMock()
        forecaster.forecast.return_value.median = np.array([[[10.]]])
        forecaster.forecast.return_value.quantile.return_value = np.array([[[10.]]])
        result = evaluate_forecast([10, 10, 12], forecaster, initial_window=2)
        self.assertEqual(result['directional_accuracy'], 0.)
        self.assertEqual(result['interval_coverage'], 0.)
        self.assertEqual(result['upper_quantile_coverage'], 0.)

    def test_interval_boundary_is_not_lost_to_rounding(self):
        forecaster = MagicMock()
        result = forecaster.forecast.return_value
        result.median = np.array([[[.25]]])
        result.quantile.side_effect = lambda q: np.array([[[.2 if q == .1 else .3]]])
        evaluation = evaluate_forecast([3., 3., .3], forecaster, initial_window=2)
        self.assertEqual(evaluation['interval_coverage'], 1.)
        self.assertEqual(evaluation['upper_quantile_coverage'], 1.)

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
