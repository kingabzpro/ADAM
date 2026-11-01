import unittest

import pandas as pd

from adam.evaluation import compare_voting
from adam.risk import RiskEvidence
from test_decisions import decision
from test_risk import EVIDENCE, LIMITS


class VotingEvaluationTests(unittest.TestCase):
    def test_compares_risk_gated_votes_on_the_same_sessions(self):
        dates = pd.date_range('2026-01-02', periods=3, tz='UTC')
        prices = pd.DataFrame({'Open': [10., 10., 12.], 'Close': [10., 12., 12.]}, index=dates)
        decisions = [[decision('market'), decision('forecast'), decision('news', 'SELL')]] * 3
        evidence = [RiskEvidence(**(EVIDENCE | {'market_time': date.to_pydatetime()})) for date in dates]
        result = compare_voting(prices, decisions, evidence, LIMITS,
                                weights={'market': 1, 'forecast': 1, 'news': 3})
        self.assertEqual(result['majority']['signals'], ['BUY'] * 3)
        self.assertEqual(result['weighted']['signals'], ['SELL'] * 3)
        self.assertAlmostEqual(result['majority']['backtest']['ending_equity'], 12000)
        self.assertAlmostEqual(result['weighted']['backtest']['ending_equity'], 10000)

    def test_missing_critical_or_risk_evidence_fails_closed(self):
        dates = pd.date_range('2026-01-02', periods=2, tz='UTC')
        prices = pd.DataFrame({'Open': [10., 10.], 'Close': [10., 12.]}, index=dates)
        decisions = [[decision('market')], [decision('market'), decision('forecast')]]
        evidence = [RiskEvidence(**(EVIDENCE | {'market_time': dates[0].to_pydatetime()})), RiskEvidence()]
        result = compare_voting(prices, decisions, evidence, LIMITS, weights={'market': 1, 'forecast': 1})
        self.assertEqual(result['majority']['signals'], ['NO_TRADE', 'NO_TRADE'])
        self.assertEqual(result['weighted']['signals'], ['NO_TRADE', 'NO_TRADE'])
        with self.assertRaises(ValueError):
            compare_voting(prices, decisions[:1], evidence, LIMITS, weights={'market': 1})

    def test_rejects_future_evidence_and_requires_fixed_weights(self):
        dates = pd.date_range('2026-01-02', periods=2, tz='UTC')
        prices = pd.DataFrame({'Open': [10., 10.], 'Close': [10., 12.]}, index=dates)
        decisions = [[decision('market'), decision('forecast')]] * 2
        future = RiskEvidence(**(EVIDENCE | {'market_time': dates[1].to_pydatetime()}))
        result = compare_voting(prices, decisions, [future, future], LIMITS,
                                weights={'market': 1, 'forecast': 1})
        self.assertEqual(result['majority']['signals'], ['NO_TRADE', 'BUY'])
        with self.assertRaises(ValueError):
            compare_voting(prices, decisions, [future, future], LIMITS, weights=None)
