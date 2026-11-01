import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from adam.fundamentals import fundamentals_evidence, fetch_fundamentals

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


class FundamentalsTests(unittest.TestCase):
    def test_extracts_fields_and_identifies_missing_and_report_age(self):
        data = {'revenueGrowth': 0.15, 'profitMargins': 0.2, 'trailingPE': 20,
                'mostRecentQuarter': int(datetime(2026, 6, 30, tzinfo=timezone.utc).timestamp())}
        result = fundamentals_evidence(data, now=NOW)
        self.assertEqual(result['metrics']['revenue_growth'], 0.15)
        self.assertIn('debt_to_equity', result['missing'])
        self.assertEqual(result['report_date'], '2026-06-30')
        self.assertEqual(result['freshness'], 'current')

    def test_missing_invalid_and_old_data_are_explicit(self):
        result = fundamentals_evidence({'trailingPE': float('nan'), 'mostRecentQuarter': 1}, now=NOW)
        self.assertIn('trailing_pe', result['missing'])
        self.assertEqual(result['freshness'], 'stale')
        self.assertEqual(fundamentals_evidence({}, now=NOW)['freshness'], 'unknown')
        self.assertEqual(fundamentals_evidence({'mostRecentQuarter': int(NOW.timestamp()) + 86400}, now=NOW)['freshness'], 'unknown')

    @patch('adam.fundamentals.yf.Ticker')
    def test_fetches_info(self, ticker):
        result = fetch_fundamentals(' aapl ')
        ticker.assert_called_once_with('AAPL')
        self.assertIs(result, ticker.return_value.get_info.return_value)
