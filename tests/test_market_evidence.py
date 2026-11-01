import unittest
from unittest.mock import AsyncMock

import pandas as pd

from adam.market import market_evidence, assess_market


def history():
    return pd.DataFrame({"Close": [100.0, 110.0, 121.0], "Volume": [1000, 2000, 3000]},
                        index=pd.date_range("2026-01-01", periods=3, tz="UTC"))


class MarketEvidenceTests(unittest.TestCase):
    def test_evidence_uses_dated_closes_and_volume(self):
        result = market_evidence(history())
        self.assertEqual(result["last_close"], 121.0)
        self.assertEqual(result["last_volume"], 3000.0)
        self.assertAlmostEqual(result["period_return"], 0.21)
        self.assertAlmostEqual(result["last_daily_return"], 0.1)
        self.assertAlmostEqual(result["annualized_volatility"], 0.0)
        self.assertEqual(result["as_of"], "2026-01-03T00:00:00+00:00")

    def test_invalid_or_insufficient_data_is_rejected(self):
        for data in [history().iloc[:2], history().assign(Close=[100, float("nan"), 121]),
                     history().assign(Close=[100, 0, 121]), history().assign(Volume=[1, 2, -1]),
                     history().iloc[::-1], history().rename(columns={"Volume": "Other"})]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    market_evidence(data)


class MarketAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_passes_evidence_to_shared_model(self):
        model = AsyncMock()
        result = await assess_market("AAPL", history(), model)
        model.assess.assert_awaited_once()
        args = model.assess.await_args.args
        self.assertEqual(args[:2], ("AAPL", "market"))
        self.assertEqual(args[2]["last_close"], 121)
        self.assertIs(result, model.assess.return_value)
