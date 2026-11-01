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
    async def test_offline_news_uses_saved_evidence_without_network(self):
        import tempfile
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock, patch
        from adam.dashboard import run_analysis
        from adam.snapshots import save_news
        from test_decisions import decision
        evidence = {'basis': 'headlines', 'headlines': [{'title': 'AAPL results',
                    'published': datetime.now(timezone.utc).isoformat(), 'sentiment': {'label': 'positive'}}],
                    'mean_scores': {'positive': .8, 'neutral': .1, 'negative': .1}}
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with tempfile.TemporaryDirectory() as directory:
            save_news(directory, 'AAPL', evidence)
            with patch('adam.dashboard.D3Client', return_value=model), \
                 patch('adam.dashboard.news_evidence', side_effect=AssertionError('network')), \
                 patch('adam.dashboard.shared_classifier', side_effect=AssertionError('model download')):
                rows = await run_analysis(['AAPL'], 1, news=True, offline=True, snapshot_directory=directory)
        self.assertIn('news: BUY', rows[0]['agents'])
        self.assertIn('[abstained]', rows[0]['agents'])
        self.assertEqual(rows[0]['evidence']['news']['source'], 'local snapshot')
        self.assertEqual(rows[0]['action'], 'NO_TRADE')

    async def test_offline_uses_cached_forecaster_without_loading_sentiment(self):
        from unittest.mock import AsyncMock, patch, sentinel
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(decisions={}, agents={}, evidence={}, errors={},
            risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('offline_snapshot',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator) as factory, \
             patch('adam.dashboard.shared_forecaster', return_value=sentinel.forecaster) as forecast, \
             patch('adam.dashboard.shared_classifier') as classifier:
            await run_analysis(['AAPL'], 1, forecast=True, news=True, offline=True, snapshot_directory='cache')
        forecast.assert_called_once_with(local_files_only=True)
        classifier.assert_not_called()
        self.assertTrue(factory.call_args.kwargs['offline'])
    async def test_ticker_concurrency_limits_active_analyses(self):
        import asyncio
        from unittest.mock import patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        active = maximum = 0
        class Coordinator:
            def __init__(self, *args, **kwargs):
                pass
            async def analyze(self, ticker, evidence, **kwargs):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(.01)
                active -= 1
                return SimpleNamespace(decisions={}, agents={}, evidence={}, errors={},
                    risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        with patch('adam.dashboard.Coordinator', Coordinator):
            await run_analysis(['AAPL', 'MSFT', 'NVDA'], 1, ticker_concurrency=2)
        self.assertEqual(maximum, 2)

    async def test_data_concurrency_is_passed_to_coordinator(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(decisions={}, agents={}, evidence={}, errors={},
            risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator) as factory:
            await run_analysis(['AAPL'], 1, data_concurrency=2, ticker_concurrency=3)
        self.assertEqual(factory.call_args.kwargs['data_concurrency'], 2)

    async def test_history_outcomes_use_ticker_history(self):
        from unittest.mock import patch, sentinel
        from adam.dashboard import saved_buy_report
        with patch('adam.dashboard.read_history', return_value=[sentinel.record]) as read, \
             patch('adam.dashboard.fetch_history', return_value=sentinel.prices) as fetch, \
             patch('adam.dashboard.evaluate_saved_buys', return_value=sentinel.report) as evaluate:
            result = await saved_buy_report('AAPL', 'history.sqlite')
        self.assertIs(result, sentinel.report)
        read.assert_called_once_with('history.sqlite', ticker='AAPL', limit=1000)
        fetch.assert_called_once_with('AAPL', '1y')
        evaluate.assert_called_once_with([sentinel.record], sentinel.prices, ticker='AAPL')

    async def test_analysis_rows_show_votes_and_risk(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        from test_decisions import decision
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(
            decisions={'market': decision('market')}, agents={'market': 'completed', 'news': 'unavailable'}, evidence={'market': {'last_close': 100}},
            risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)), errors={})
        with patch('adam.dashboard.Coordinator', return_value=coordinator):
            rows = await run_analysis(['AAPL'], inference_concurrency=1)
        self.assertEqual(rows[0]['ticker'], 'AAPL')
        self.assertIn('market: BUY', rows[0]['agents'])
        self.assertIn('news: unavailable', rows[0]['agents'])
        self.assertEqual(rows[0]['action'], 'NO_TRADE')
        self.assertIn('risk_limits', rows[0]['risk'])
        self.assertEqual(rows[0]['evidence']['market']['last_close'], 100)

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
        coordinator.analyze.return_value = SimpleNamespace(decisions={}, agents={}, evidence={}, errors={}, risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('volatility',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator) as factory:
            await run_analysis(['AAPL', 'MSFT'], 1, LIMITS, {'AAPL': RiskEvidence(volatility=0.2)})
        self.assertEqual(factory.call_args.args[1], LIMITS)
        calls = {call.args[0]: call.args[1] for call in coordinator.analyze.await_args_list}
        self.assertEqual(calls['AAPL'].volatility, 0.2)
        self.assertIsNone(calls['MSFT'].volatility)

    async def test_forecaster_can_be_used_in_analysis(self):
        from unittest.mock import AsyncMock, patch, sentinel
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(decisions={}, agents={}, evidence={}, errors={}, risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator) as factory, patch('adam.dashboard.shared_forecaster', return_value=sentinel.forecaster):
            await run_analysis(['AAPL'], 1, forecast=True)
        self.assertIs(factory.call_args.kwargs['forecaster'], sentinel.forecaster)

    def test_agent_score_format_includes_all_scores(self):
        from adam.dashboard import format_agent
        from test_decisions import decision
        formatted = format_agent('market', 'completed', decision('market'))
        self.assertIn('market: BUY', formatted)
        self.assertIn('BUY 70.0%', formatted)
        self.assertIn('HOLD 20.0%', formatted)
        self.assertIn('abstained', format_agent('market', 'completed', decision('market', reliable=False)))
        self.assertEqual(format_agent('news', 'unavailable', None), 'news: unavailable')

    async def test_analysis_path_uses_local_model_and_fails_closed(self):
        import httpx
        from datetime import datetime, timezone
        from unittest.mock import patch
        from adam.dashboard import run_analysis
        from test_technical import history
        calls = []
        def respond(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={'answers': {'signal': {'choice': 'BUY', 'probabilities': {'BUY': 0.7, 'HOLD': 0.2, 'SELL': 0.1}}}})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        info = {'revenueGrowth': 0.1, 'mostRecentQuarter': int(datetime.now(timezone.utc).timestamp())}
        with patch('adam.dashboard.httpx.AsyncClient', return_value=client), patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)), patch('adam.coordinator.fetch_fundamentals', return_value=info):
            rows = await run_analysis(['AAPL'], 2)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(path == '/v1/systemone' for path in calls))
        self.assertIn('market: BUY', rows[0]['agents'])
        self.assertEqual(rows[0]['action'], 'NO_TRADE')

    async def test_dashboard_forwards_progress_events(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        events = []
        coordinator = AsyncMock()
        async def analyze(ticker, evidence, *, on_event):
            on_event({'ticker': ticker, 'stage': 'market', 'status': 'completed', 'time': 'now'})
            return SimpleNamespace(decisions={}, agents={}, evidence={}, errors={}, risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        coordinator.analyze.side_effect = analyze
        with patch('adam.dashboard.Coordinator', return_value=coordinator):
            await run_analysis(['AAPL'], 1, on_event=events.append)
        self.assertEqual(events[0]['stage'], 'market')
        self.assertEqual(events[0]['status'], 'completed')

    async def test_news_option_supplies_news_provider(self):
        from unittest.mock import AsyncMock, patch, sentinel
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(decisions={}, agents={}, evidence={}, errors={}, risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator) as factory, patch('adam.dashboard.shared_classifier', return_value=sentinel.classifier):
            await run_analysis(['AAPL'], 1, news=True)
        self.assertTrue(callable(factory.call_args.kwargs['news_provider']))

    async def test_analysis_can_be_saved_with_risk_context(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.history import read_history
        from adam.risk import RiskEvidence, RiskResult
        from test_decisions import decision
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(ticker='AAPL', decisions={'market': decision('market')}, agents={'market': 'completed'}, evidence={}, errors={}, timeline=[], risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'history.sqlite'
            with patch('adam.dashboard.Coordinator', return_value=coordinator):
                rows = await run_analysis(['AAPL'], 1, evidence={'AAPL': RiskEvidence(volatility=0.2)}, history_path=path)
            saved = read_history(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]['risk_evidence']['volatility'], 0.2)
            self.assertNotIn('history:', rows[0]['errors'])

    async def test_history_write_failure_is_visible_without_losing_result(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from adam.dashboard import run_analysis
        from adam.risk import RiskResult
        coordinator = AsyncMock()
        coordinator.analyze.return_value = SimpleNamespace(ticker='AAPL', decisions={}, agents={}, evidence={}, errors={}, timeline=[], risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('risk_limits',)))
        with patch('adam.dashboard.Coordinator', return_value=coordinator), patch('adam.dashboard.record_analysis', side_effect=PermissionError('read-only')):
            rows = await run_analysis(['AAPL'], 1, history_path='ignored')
        self.assertEqual(rows[0]['action'], 'NO_TRADE')
        self.assertIn('history: read-only', rows[0]['errors'])
