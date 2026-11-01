import unittest

import pandas as pd

from adam.dashboard import market_rows, parse_tickers


class DashboardTests(unittest.TestCase):
    def test_parse_tickers_deduplicates_and_normalizes(self):
        self.assertEqual(parse_tickers(" aapl, MSFT, AAPL ,,"), ["AAPL", "MSFT"])

    def test_market_rows_show_latest_close_and_failures(self):
        history = pd.DataFrame({"Close": [100.0, 101.5]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
        rows = market_rows(["AAPL", "BAD", "EMPTY"], [history, RuntimeError("offline"), pd.DataFrame()])
        self.assertEqual(rows[0], {"ticker": "AAPL", "date": "2026-01-02", "close": "101.50", "status": "Daily close"})
        self.assertEqual(rows[1]["status"], "Data unavailable")
        self.assertEqual(rows[2]["status"], "No price history")
