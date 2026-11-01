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

class AnalysisDisplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_rows_show_votes_and_risk(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        from test_decisions import decision
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(
            decisions={'market': decision('market')}, agents={'market': 'completed', 'news': 'unavailable'},
            risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)), errors={})
        with patch('adam.dashboard.Coordinator', return_value=coordinator):
            rows = await run_analysis(['AAPL'], inference_concurrency=1)
        self.assertEqual(rows[0]['ticker'], 'AAPL')
        self.assertIn('market: BUY', rows[0]['agents'])
        self.assertIn('news: unavailable', rows[0]['agents'])
        self.assertEqual(rows[0]['action'], 'NO_TRADE')
        self.assertIn('risk_limits', rows[0]['risk'])

class RiskInputTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_explicit_per_ticker_evidence(self):
        import json
        from adam.dashboard import parse_risk_input
        from test_risk import LIMITS
        data = {'limits': LIMITS.model_dump(), 'tickers': {'AAPL': {'volatility': 0.2}}}
        limits, evidence = parse_risk_input(json.dumps(data))
        self.assertEqual(limits.max_volatility, 0.5)
        self.assertEqual(evidence['AAPL'].volatility, 0.2)
        self.assertIsNone(evidence['AAPL'].market_time)

    def test_rejects_malformed_risk_configuration(self):
        from adam.dashboard import parse_risk_input
        self.assertEqual(parse_risk_input(''), (None, {}))
        for value in ('[]', '{"limits": {}, "tickers": {}}', '{"tickers": {}}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_risk_input(value)

    async def test_passes_each_tickers_own_risk_evidence(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskEvidence, RiskResult
        from test_risk import LIMITS
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(decisions={}, agents={}, errors={}, risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('volatility',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator) as factory:
            await run_analysis(['AAPL', 'MSFT'], 1, LIMITS, {'AAPL': RiskEvidence(volatility=0.2)})
        self.assertEqual(factory.call_args.args[1], LIMITS)
        calls = {call.args[0]: call.args[1] for call in coordinator.analyze.await_args_list}
        self.assertEqual(calls['AAPL'].volatility, 0.2)
        self.assertIsNone(calls['MSFT'].volatility)
