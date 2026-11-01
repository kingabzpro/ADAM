import asyncio
import unittest
from unittest.mock import patch

from adam.market import fetch_many


class MultiTickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_results_and_per_ticker_errors(self):
        def fetch(ticker):
            if ticker == "BAD":
                raise RuntimeError("unavailable")
            return ticker + " data"

        with patch("adam.market.fetch_history", side_effect=fetch):
            results = await fetch_many(["AAPL", "BAD"], concurrency=2)
        self.assertEqual(results[0], "AAPL data")
        self.assertIsInstance(results[1], RuntimeError)

    async def test_rejects_invalid_concurrency(self):
        with self.assertRaises(ValueError):
            await fetch_many(["AAPL"], concurrency=0)
