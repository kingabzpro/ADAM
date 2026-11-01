import unittest

import pandas as pd

from adam.evaluation import compare_rule_baseline
from adam.risk import RiskEvidence
from test_decisions import decision
from test_risk import EVIDENCE, LIMITS


class RuleEvaluationTests(unittest.TestCase):
    def test_baseline_uses_only_past_closes_and_same_risk_gate(self):
        dates = pd.date_range('2026-01-02', periods=5, tz='UTC')
        prices = pd.DataFrame({'Open': [10., 10., 11., 12., 8.],
                               'Close': [10., 11., 12., 8., 9.]}, index=dates)
        votes = [[decision('market'), decision('forecast')]] * len(dates)
        evidence = [RiskEvidence(**(EVIDENCE | {'market_time': t.to_pydatetime()})) for t in dates]
        result = compare_rule_baseline(prices, votes, evidence, LIMITS, window=2)
        self.assertEqual(result['rule']['signals'], ['NO_TRADE', 'BUY', 'BUY', 'SELL', 'BUY'])
        self.assertEqual(result['model']['signals'], ['NO_TRADE', 'BUY', 'BUY', 'BUY', 'BUY'])
        self.assertEqual(result['rule']['backtest']['trades'][0]['date'], '2026-01-04')
        self.assertEqual(result['rule']['backtest']['trades'][1]['date'], '2026-01-06')
        changed_future = prices.copy()
        changed_future.loc[dates[4], 'Close'] = 1000
        changed = compare_rule_baseline(changed_future, votes, evidence, LIMITS, window=2)
        self.assertEqual(changed['rule']['signals'][:4], result['rule']['signals'][:4])
        blocked = evidence.copy()
        blocked[1] = RiskEvidence(**(EVIDENCE | {'market_time': dates[1].to_pydatetime(), 'downside': 0.9}))
        result = compare_rule_baseline(prices, votes, blocked, LIMITS, window=2)
        self.assertEqual(result['rule']['signals'][1], 'NO_TRADE')
        self.assertEqual(result['model']['signals'][1], 'NO_TRADE')

    def test_invalid_window_is_rejected(self):
        prices = pd.DataFrame({'Open': [10., 11.], 'Close': [10., 11.]},
                              index=pd.date_range('2026-01-01', periods=2, tz='UTC'))
        for window in (True, 1, 3):
            with self.subTest(window=window), self.assertRaises(ValueError):
                compare_rule_baseline(prices, [[], []], [RiskEvidence()] * 2, LIMITS, window=window)
