import unittest
from datetime import datetime, timezone

import pandas as pd

from adam.evaluation import evaluate_saved_buys


class HistoryEvaluationTests(unittest.TestCase):
    def test_uses_later_completed_sessions_and_leaves_pending_visible(self):
        dates = pd.date_range('2026-01-02', periods=4, tz='America/New_York')
        prices = pd.DataFrame({'Open': [10., 10., 12., 14.], 'Close': [10., 12., 13., 16.]}, index=dates)
        records = [
            {'id': 1, 'ticker': 'AAPL', 'created_at': '2026-01-02T23:00:00+00:00', 'risk': {'status': 'pass', 'action': 'BUY'}},
            {'id': 2, 'ticker': 'AAPL', 'created_at': '2026-01-04T23:00:00+00:00', 'risk': {'status': 'warning', 'action': 'BUY'}},
            {'id': 3, 'ticker': 'AAPL', 'created_at': '2026-01-02T23:00:00+00:00', 'risk': {'status': 'fail', 'action': 'NO_TRADE'}},
        ]
        result = evaluate_saved_buys(records, prices, ticker='AAPL', horizon=2, cost_bps=100,
                                     now=datetime(2026, 1, 5, 20, tzinfo=timezone.utc))
        self.assertEqual([row['status'] for row in result['outcomes']], ['completed', 'pending'])
        self.assertEqual(result['outcomes'][0]['entry_date'], '2026-01-03')
        self.assertEqual(result['outcomes'][0]['exit_date'], '2026-01-04')
        self.assertAlmostEqual(result['outcomes'][0]['return'], 13 / 10 * .99 / 1.01 - 1)
        self.assertEqual(result['completed_count'], 1)
        self.assertEqual(result['win_rate'], 1)

    def test_invalid_and_mixed_inputs_are_rejected(self):
        prices = pd.DataFrame({'Open': [10., 11.], 'Close': [10., 11.]},
                              index=pd.date_range('2026-01-01', periods=2, tz='UTC'))
        record = {'id': 1, 'ticker': 'MSFT', 'created_at': '2026-01-01T00:00:00+00:00', 'risk': {'status': 'pass', 'action': 'BUY'}}
        with self.assertRaises(ValueError):
            evaluate_saved_buys([record], prices, ticker='AAPL')
        with self.assertRaises(ValueError):
            evaluate_saved_buys([], prices, ticker='AAPL', horizon=True)
        with self.assertRaises(ValueError):
            evaluate_saved_buys([], prices.assign(Open=0), ticker='AAPL')

    def test_older_than_available_history_is_not_treated_as_next_session(self):
        prices = pd.DataFrame({'Open': [10., 11.], 'Close': [10., 11.]},
                              index=pd.date_range('2026-01-02', periods=2, tz='UTC'))
        record = {'id': 1, 'ticker': 'AAPL', 'created_at': '2026-01-01T00:00:00+00:00', 'risk': {'status': 'pass', 'action': 'BUY'}}
        result = evaluate_saved_buys([record], prices, ticker='AAPL', horizon=1,
                                     now=datetime(2026, 1, 5, tzinfo=timezone.utc))
        self.assertEqual(result['outcomes'][0]['status'], 'history_unavailable')
        self.assertIsNone(result['mean_return'])
