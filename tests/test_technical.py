import unittest
from unittest.mock import AsyncMock

import pandas as pd

from adam.technical import technical_evidence, assess_technical


def history(values):
    return pd.DataFrame({"High": [v + 1 for v in values], "Low": [v - 1 for v in values],
                         "Close": values, "Volume": [1000] * len(values)},
                        index=pd.date_range("2026-01-01", periods=len(values), tz="UTC"))


class TechnicalTests(unittest.TestCase):
    def test_flat_series_indicators(self):
        result = technical_evidence(history([100.] * 40))
        self.assertEqual(result["rsi_14"], 50)
        self.assertAlmostEqual(result["sma_20"], 100)
        self.assertAlmostEqual(result["macd"], 0)
        self.assertAlmostEqual(result["atr_14"], 2)
        self.assertAlmostEqual(result["bollinger_upper"], 100)
        self.assertAlmostEqual(result["volume_ratio_20"], 1)
        self.assertEqual(result["support_20"], 99)
        self.assertEqual(result["resistance_20"], 101)

    def test_increasing_series_and_validation(self):
        result = technical_evidence(history(list(range(100, 140))))
        self.assertEqual(result["rsi_14"], 100)
        self.assertGreater(result["macd"], 0)
        self.assertAlmostEqual(result["momentum_10"], 139 / 129 - 1)
        for frame in [history([100.] * 10), history([100.] * 40).assign(Close=float('nan')),
                      history([100.] * 40).iloc[::-1], history([100.] * 40).assign(High=90)]:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                technical_evidence(frame)


class TechnicalAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_evidence_to_shared_model(self):
        model = AsyncMock()
        await assess_technical('AAPL', history([100.] * 40), model)
        self.assertEqual(model.assess.await_args.args[:2], ('AAPL', 'technical'))
