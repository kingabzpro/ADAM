import unittest

import pandas as pd

from adam.market import compare_market


def prices(values, start='2026-01-01'):
    return pd.DataFrame({'Close': values}, index=pd.date_range(start, periods=len(values), tz='UTC'))


class MarketComparisonTests(unittest.TestCase):
    def test_uses_matching_dates_and_reports_benchmark_trend(self):
        result = compare_market(prices(list(range(100, 125))), prices([200.] * 25))
        self.assertEqual(result['benchmark'], 'SPY')
        self.assertAlmostEqual(result['relative_return'], 0.24)
        self.assertEqual(result['benchmark_vs_sma_20'], 'at')
        self.assertEqual(result['aligned_observations'], 25)

    def test_rejects_missing_or_misaligned_latest_data(self):
        for asset, benchmark in [(prices([10.] * 19), prices([10.] * 19)),
                                 (prices([10.] * 25), prices([10.] * 25, start='2025-12-31')),
                                 (prices([10.] * 25), prices([float('nan')] * 25))]:
            with self.subTest(), self.assertRaises(ValueError):
                compare_market(asset, benchmark)
