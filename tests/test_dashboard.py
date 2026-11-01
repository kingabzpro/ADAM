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
