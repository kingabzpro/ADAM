import unittest
from datetime import datetime, timedelta, timezone

from adam.decisions import consensus
from adam.risk import RiskEvidence, RiskLimits, evaluate_risk
from test_decisions import decision

NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)
LIMITS = RiskLimits(max_age_seconds=3600, max_volatility=0.5, max_downside=0.2,
                    max_position_fraction=0.1, max_concentration=0.3,
                    min_daily_dollar_volume=1000000, max_correlated_exposure=0.4,
                    min_reward_risk=2, max_transaction_cost_fraction=0.01,
                    max_loss_fraction=0.05)
EVIDENCE = dict(market_time=NOW, volatility=0.2, downside=0.1, position_fraction=0.05,
                concentration=0.2, daily_dollar_volume=2000000, correlated_exposure=0.3,
                reward_risk=3, transaction_cost_fraction=0.005, loss_fraction=0.02)


class RiskTests(unittest.TestCase):
    def evaluate(self, evidence, decisions=None):
        return evaluate_risk(consensus(decisions or [decision("market")]), evidence, LIMITS, now=NOW)

    def test_pass_and_disagreement_warning(self):
        evidence = RiskEvidence(**EVIDENCE)
        self.assertEqual(self.evaluate(evidence).action, "BUY")
        result = self.evaluate(evidence, [decision("market"), decision("news", "SELL")])
        self.assertEqual((result.status, result.action), ("warning", "HOLD"))

    def test_each_limit_violation_fails_closed(self):
        failures = {"market_time": NOW - timedelta(seconds=3601), "volatility": 0.6,
                    "downside": 0.3, "position_fraction": 0.2, "concentration": 0.4,
                    "daily_dollar_volume": 999999, "correlated_exposure": 0.5,
                    "reward_risk": 1, "transaction_cost_fraction": 0.02, "loss_fraction": 0.06}
        for key, value in failures.items():
            with self.subTest(key=key):
                result = self.evaluate(RiskEvidence(**(EVIDENCE | {key: value})))
                self.assertEqual((result.status, result.action), ("fail", "NO_TRADE"))
                self.assertIn(key, result.reasons)

    def test_missing_and_future_data_fail_closed(self):
        result = self.evaluate(RiskEvidence(**(EVIDENCE | {"volatility": None})))
        self.assertEqual((result.status, result.action), ("insufficient_evidence", "NO_TRADE"))
        self.assertEqual(self.evaluate(RiskEvidence(**(EVIDENCE | {"market_time": NOW + timedelta(seconds=1)}))).action, "NO_TRADE")
        self.assertEqual(evaluate_risk(consensus([]), RiskEvidence(**EVIDENCE), LIMITS, now=NOW).action, "NO_TRADE")

    def test_invalid_numbers_rejected(self):
        with self.assertRaises(ValueError):
            RiskEvidence(**(EVIDENCE | {"volatility": float("nan")}))
        with self.assertRaises(ValueError):
            RiskLimits(**(LIMITS.model_dump() | {"max_age_seconds": -1}))

    def test_decision_path_cannot_skip_risk(self):
        from adam.risk import decide
        result = decide([decision("market")], RiskEvidence(), LIMITS, now=NOW)
        self.assertEqual((result.status, result.action), ("insufficient_evidence", "NO_TRADE"))
        result = decide([decision("market")], RiskEvidence(**EVIDENCE), LIMITS, now=NOW)
        self.assertEqual((result.status, result.action), ("pass", "BUY"))
