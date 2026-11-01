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
    def test_observed_market_and_forecast_evidence_cannot_be_made_safer_by_user_input(self):
        from adam.risk import derive_risk_evidence
        data = RiskEvidence(**(EVIDENCE | {'volatility': .1, 'downside': .02, 'daily_dollar_volume': 5000000}))
        market = {'as_of': (NOW - timedelta(hours=1)).isoformat(), 'annualized_volatility': .6,
                  'last_close': 100, 'last_volume': 10000}
        forecast = {'lower_returns': [-.1, -.3, -.2]}
        actual = derive_risk_evidence(data, market, forecast)
        self.assertEqual(actual.market_time, NOW - timedelta(hours=1))
        self.assertEqual(actual.volatility, .6)
        self.assertEqual(actual.downside, .3)
        self.assertEqual(actual.daily_dollar_volume, 1000000)
        self.assertEqual(self.evaluate(actual).action, 'NO_TRADE')

    def test_derived_values_fill_missing_and_keep_more_conservative_user_values(self):
        from adam.risk import derive_risk_evidence
        market = {'as_of': NOW.isoformat(), 'annualized_volatility': .2, 'last_close': 100, 'last_volume': 10000}
        actual = derive_risk_evidence(RiskEvidence(volatility=.4, downside=.5, daily_dollar_volume=500), market,
                                      {'lower_returns': [.1, .2]})
        self.assertEqual((actual.market_time, actual.volatility, actual.downside, actual.daily_dollar_volume), (NOW, .4, .5, 500))
        actual = derive_risk_evidence(RiskEvidence(), market, {'lower_returns': [.1, -.2]})
        self.assertEqual((actual.market_time, actual.volatility, actual.downside, actual.daily_dollar_volume), (NOW, .2, .2, 1000000))

    def test_invalid_observations_cannot_be_replaced_by_optimistic_user_values(self):
        from adam.risk import derive_risk_evidence
        market = {'as_of': 'bad', 'annualized_volatility': float('nan'), 'last_close': 100, 'last_volume': -1}
        actual = derive_risk_evidence(RiskEvidence(**EVIDENCE), market, {'lower_returns': [float('nan')]})
        self.assertIsNone(actual.market_time)
        self.assertIsNone(actual.volatility)
        self.assertIsNone(actual.daily_dollar_volume)
        self.assertIsNone(actual.downside)
        self.assertEqual(self.evaluate(actual).action, 'NO_TRADE')

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

    def test_without_portfolio_only_market_checks_apply(self):
        market_only = RiskEvidence(market_time=NOW, volatility=0.2, downside=0.1, daily_dollar_volume=2000000)
        self.assertEqual(self.evaluate(market_only).action, "BUY")
        result = self.evaluate(market_only, [decision("market"), decision("news", "SELL")])
        self.assertEqual((result.status, result.action), ("warning", "HOLD"))
        stale = market_only.model_copy(update={"market_time": NOW - timedelta(seconds=3601)})
        self.assertEqual(self.evaluate(stale).status, "fail")
        partial = market_only.model_copy(update={"position_fraction": 0.05})
        result = self.evaluate(partial)
        self.assertEqual((result.status, result.action), ("insufficient_evidence", "NO_TRADE"))
        self.assertIn("concentration", result.reasons)
