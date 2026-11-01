import unittest
from unittest.mock import patch

from adam.market import fetch_history


class MarketTests(unittest.TestCase):
    @patch("adam.market.yf.Ticker")
    def test_fetches_price_history(self, ticker):
        result = fetch_history(" aapl ")
        ticker.assert_called_once_with("AAPL")
        ticker.return_value.history.assert_called_once_with(period="1mo", interval="1d", raise_errors=True)
        self.assertIs(result, ticker.return_value.history.return_value)

    def test_rejects_blank_ticker(self):
        with self.assertRaises(ValueError):
            fetch_history(" ")
