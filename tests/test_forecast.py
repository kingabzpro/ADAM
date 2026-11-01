import unittest
from unittest.mock import MagicMock

import numpy as np

from adam.forecast import forecast_evidence


class ForecastTests(unittest.TestCase):
    def test_quantiles_and_returns_for_multiple_horizons(self):
        model = MagicMock()
        output = model.forecast.return_value
        output.median = np.array([[[110., 120.]]])
        output.quantile.side_effect = lambda q: np.array([[[90., 95.]]]) if q == 0.1 else np.array([[[120., 140.]]])
        result = forecast_evidence([90, 100], 2, model)
        np.testing.assert_array_equal(model.forecast.call_args.args[0], [90, 100])
        self.assertEqual(result["horizons"], [1, 2])
        self.assertAlmostEqual(result["median_returns"][-1], 0.2)
        self.assertAlmostEqual(result["lower_returns"][-1], -0.05)
        self.assertAlmostEqual(result["upper_returns"][-1], 0.4)

    def test_rejects_invalid_input_and_output(self):
        model = MagicMock()
        for values, horizon in [([], 1), ([1, 0], 1), ([1, float("nan")], 1), ([1, 2], 0)]:
            with self.subTest(values=values, horizon=horizon), self.assertRaises(ValueError):
                forecast_evidence(values, horizon, model)
        model.forecast.return_value.median = np.array([[[float("nan")]]])
        with self.assertRaises(ValueError):
            forecast_evidence([1, 2], 1, model)
