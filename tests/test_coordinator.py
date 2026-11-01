import unittest
from unittest.mock import AsyncMock, patch

from adam.coordinator import Coordinator
from adam.risk import RiskEvidence
from test_decisions import decision
from test_risk import LIMITS
from test_technical import history


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patch_fundamentals = patch('adam.coordinator.fetch_fundamentals', side_effect=RuntimeError('unavailable'))
        self.patch_fundamentals.start()
        self.addCleanup(self.patch_fundamentals.stop)

    async def test_offline_analysis_uses_snapshots_and_never_approves_trade(self):
        import tempfile
        from datetime import datetime, timezone
        from adam.snapshots import save_prices, save_fundamentals
        from test_risk import EVIDENCE
        prices = history([100.] * 40)
        prices['Open'] = 100.
        now = datetime.now(timezone.utc)
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with tempfile.TemporaryDirectory() as directory:
            for ticker in ('AAPL', 'SPY'):
                save_prices(directory, ticker, prices)
            save_fundamentals(directory, 'AAPL', {'revenueGrowth': .1, 'mostRecentQuarter': now.timestamp()})
            with patch('adam.coordinator.fetch_history', side_effect=AssertionError('network')), \
                 patch('adam.coordinator.fetch_fundamentals', side_effect=AssertionError('network')):
                result = await Coordinator(model, LIMITS, offline=True, snapshot_directory=directory).analyze('AAPL', RiskEvidence(**(EVIDENCE | {'market_time': now})))
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertIn('offline_snapshot', result.risk.reasons)
        self.assertEqual(result.evidence['market']['source'], 'local snapshot')
        self.assertEqual(result.evidence['fundamentals']['source'], 'local snapshot')

    async def test_offline_missing_snapshots_do_not_fall_back_to_network(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory, \
             patch('adam.coordinator.fetch_history', side_effect=AssertionError('network')), \
             patch('adam.coordinator.fetch_fundamentals', side_effect=AssertionError('network')):
            result = await Coordinator(AsyncMock(), None, offline=True, snapshot_directory=directory).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertEqual(result.agents['market'], 'failed')
        self.assertIn('market_data', result.errors)

    async def test_online_analysis_saves_data_for_future_offline_use(self):
        import tempfile
        from datetime import datetime, timezone
        from adam.snapshots import read_prices, read_fundamentals
        prices = history([100.] * 40)
        prices['Open'] = 100.
        info = {'revenueGrowth': .1, 'mostRecentQuarter': datetime.now(timezone.utc).timestamp()}
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with tempfile.TemporaryDirectory() as directory, \
             patch('adam.coordinator.fetch_history', return_value=prices), \
             patch('adam.coordinator.fetch_fundamentals', return_value=info):
            await Coordinator(model, None, snapshot_directory=directory).analyze('AAPL', RiskEvidence())
            self.assertEqual(len(read_prices(directory, 'AAPL')[0]), 40)
            self.assertEqual(len(read_prices(directory, 'SPY')[0]), 40)
            self.assertEqual(read_fundamentals(directory, 'AAPL')[0]['revenueGrowth'], .1)

    async def test_snapshot_write_failure_is_reported_without_losing_analysis(self):
        import tempfile
        from datetime import datetime, timezone
        prices = history([100.] * 40)
        prices['Open'] = 100.
        info = {'revenueGrowth': .1, 'mostRecentQuarter': datetime.now(timezone.utc).timestamp()}
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with tempfile.TemporaryDirectory() as directory, \
             patch('adam.coordinator.fetch_history', return_value=prices), \
             patch('adam.coordinator.fetch_fundamentals', return_value=info), \
             patch('adam.snapshots.save_prices', side_effect=PermissionError('read only')):
            with self.assertLogs(level='ERROR'):
                result = await Coordinator(model, None, snapshot_directory=directory).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['market'], 'completed')
        self.assertIn('read only', result.errors['snapshot_prices'])

    async def test_data_requests_share_inflight_work_and_survive_one_cancellation(self):
        import asyncio
        from unittest.mock import AsyncMock
        ready, release = asyncio.Event(), asyncio.Event()
        async def fetch(*args):
            ready.set()
            await release.wait()
            return 'data'
        coordinator = Coordinator(AsyncMock(), None)
        with patch('adam.coordinator.asyncio.to_thread', side_effect=fetch) as run:
            first = asyncio.create_task(coordinator._fetch_data(('prices', 'AAPL'), object()))
            await ready.wait()
            second = asyncio.create_task(coordinator._fetch_data(('prices', 'AAPL'), object()))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual(await second, 'data')
            self.assertEqual(run.call_count, 1)
            self.assertEqual(await coordinator._fetch_data(('prices', 'AAPL'), object()), 'data')
            self.assertEqual(run.call_count, 2)

    async def test_data_workers_are_bounded_and_failed_work_retries(self):
        import asyncio
        from unittest.mock import AsyncMock
        active = maximum = 0
        async def fetch(function, name):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(.01)
                if name == 'bad':
                    raise RuntimeError('offline')
                return name
            finally:
                active -= 1
        coordinator = Coordinator(AsyncMock(), None, data_concurrency=2)
        with patch('adam.coordinator.asyncio.to_thread', side_effect=fetch) as run:
            results = await asyncio.gather(*(coordinator._fetch_data(('prices', name), object(), name)
                                             for name in ['a', 'b', 'c', 'bad']), return_exceptions=True)
            self.assertEqual(maximum, 2)
            self.assertIsInstance(results[-1], RuntimeError)
            with self.assertRaises(RuntimeError):
                await coordinator._fetch_data(('prices', 'bad'), object(), 'bad')
            self.assertEqual(run.call_count, 5)

    async def test_runs_available_agents_and_applies_risk_gate(self):
        model = AsyncMock()
        async def assess(ticker, agent, evidence):
            return decision(agent)
        model.assess.side_effect = assess
        coordinator = Coordinator(model, LIMITS, inference_concurrency=2)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            result = await coordinator.analyze('AAPL', RiskEvidence())
        self.assertEqual(set(result.decisions), {'market', 'technical'})
        self.assertEqual(result.agents['forecast'], 'unavailable')
        self.assertEqual(result.agents['news'], 'unavailable')
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertEqual(result.risk.status, 'insufficient_evidence')

    async def test_one_agent_failure_is_isolated(self):
        model = AsyncMock()
        async def assess(ticker, agent, evidence):
            if agent == 'technical':
                raise RuntimeError('model unavailable')
            return decision(agent)
        model.assess.side_effect = assess
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            result = await Coordinator(model, LIMITS).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['technical'], 'failed')
        self.assertEqual(result.errors['technical'], 'model unavailable')
        self.assertEqual(result.agents['market'], 'completed')
        self.assertEqual(result.risk.action, 'NO_TRADE')

    async def test_data_failure_returns_no_trade(self):
        model = AsyncMock()
        with patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, LIMITS).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['market'], 'failed')
        self.assertEqual(result.errors['market_data'], 'offline')
        self.assertEqual(result.risk.action, 'NO_TRADE')
        model.assess.assert_not_awaited()

    async def test_fundamentals_runs_when_price_collection_fails(self):
        from datetime import datetime, timezone
        model = AsyncMock()
        model.assess.return_value = decision('fundamentals')
        info = {'revenueGrowth': 0.1, 'mostRecentQuarter': int(datetime.now(timezone.utc).timestamp())}
        with patch('adam.coordinator.fetch_fundamentals', return_value=info), patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, LIMITS).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['fundamentals'], 'completed')
        self.assertEqual(result.agents['market'], 'failed')
        self.assertEqual(result.risk.action, 'NO_TRADE')

    async def test_stale_fundamentals_are_not_sent_to_model(self):
        model = AsyncMock()
        with patch('adam.coordinator.fetch_fundamentals', return_value={'revenueGrowth': 0.1, 'mostRecentQuarter': 1}), patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, LIMITS).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['fundamentals'], 'failed')
        model.assess.assert_not_awaited()

    async def test_missing_risk_limits_never_approves_vote(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.risk.status, 'insufficient_evidence')
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertIn('risk_limits', result.risk.reasons)

    async def test_missing_forecast_cannot_approve_even_with_complete_risk_data(self):
        from test_risk import EVIDENCE, NOW
        from unittest.mock import patch
        from adam.risk import RiskEvidence
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)), patch('adam.risk.datetime') as clock:
            clock.now.return_value = NOW
            result = await Coordinator(model, LIMITS).analyze('AAPL', RiskEvidence(**EVIDENCE))
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertIn('forecast_decision', result.risk.reasons)

    async def test_emits_progress_and_records_risk_outcome(self):
        events = []
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence(), on_event=events.append)
        self.assertEqual(events, result.timeline)
        self.assertTrue(any(event['stage'] == 'market' and event['status'] == 'completed' for event in events))
        self.assertTrue(any(event['stage'] == 'fundamentals' and event['status'] == 'failed' for event in events))
        self.assertEqual(events[-1]['stage'], 'risk')
        self.assertEqual(events[-1]['status'], 'NO_TRADE')
        self.assertTrue(all(event['ticker'] == 'AAPL' and event['time'].endswith('+00:00') for event in events))

    async def test_bad_event_callback_does_not_change_risk_result(self):
        def broken(event):
            raise RuntimeError('viewer disconnected')
        with patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            with self.assertLogs(level='ERROR'):
                result = await Coordinator(AsyncMock(), None).analyze('AAPL', RiskEvidence(), on_event=broken)
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertEqual(result.timeline[-1]['stage'], 'risk')

    async def test_news_provider_contributes_one_category_vote(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        provider = AsyncMock(return_value={'basis': 'headlines', 'headlines': [{'title': 'news'}], 'mean_scores': {'positive': 1}})
        with patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, None, news_provider=provider).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['news'], 'completed')
        self.assertEqual(result.decisions['news'].agent, 'news')
        self.assertEqual(result.risk.action, 'NO_TRADE')

    async def test_empty_news_abstains(self):
        model = AsyncMock()
        provider = AsyncMock(return_value={'basis': 'headlines', 'headlines': [], 'mean_scores': None})
        with patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, None, news_provider=provider).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['news'], 'failed')
        model.assess.assert_not_awaited()

    async def test_article_news_contributes_one_category_vote(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        article = {'title': 'AAPL results', 'summary': 'Revenue grew', 'published': '2026-10-01', 'sentiment': {'label': 'positive'}}
        provider = AsyncMock(return_value={'basis': 'articles', 'articles': [article], 'mean_scores': {'positive': .8, 'neutral': .1, 'negative': .1}})
        with patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, None, news_provider=provider).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['news'], 'completed')
        self.assertEqual(result.decisions['news'].agent, 'news')

    async def test_evidence_is_preserved_even_when_model_fails(self):
        model = AsyncMock()
        model.assess.side_effect = RuntimeError('model offline')
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.agents['market'], 'failed')
        self.assertEqual(result.evidence['market']['last_close'], 100.0)
        self.assertEqual(result.evidence['technical']['sma_20'], 100.0)

    async def test_market_agent_includes_benchmark_comparison(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        prices = history([float(n) for n in range(100, 140)])
        with patch('adam.coordinator.fetch_history', return_value=prices) as fetch:
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence())
        self.assertEqual(result.evidence['market']['comparison']['benchmark'], 'SPY')
        self.assertTrue(any(call.args[0] == 'SPY' for call in fetch.call_args_list))

    async def test_unavailable_benchmark_is_explicit(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        def fetch(ticker, period):
            if ticker == 'SPY':
                raise RuntimeError('benchmark unavailable')
            return history([100.] * 40)
        with patch('adam.coordinator.fetch_history', side_effect=fetch):
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence())
        self.assertIsNone(result.evidence['market']['comparison'])
        self.assertEqual(result.evidence['market']['comparison_status'], 'unavailable')

    async def test_missing_market_comparison_abstains_from_vote(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        def fetch(ticker, period):
            if ticker == 'SPY':
                raise RuntimeError('offline')
            return history([100.] * 40)
        with patch('adam.coordinator.fetch_history', side_effect=fetch):
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence())
        self.assertFalse(result.decisions['market'].reliable)
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertIn('market_decision', result.risk.reasons)

    async def test_concurrent_tickers_share_one_benchmark_download(self):
        import asyncio
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent).model_copy(update={'ticker': ticker})
        frame = history([100.] * 40)
        coordinator = Coordinator(model, None, inference_concurrency=2)
        with patch('adam.coordinator.fetch_history', return_value=frame) as fetch:
            results = await asyncio.gather(coordinator.analyze('AAPL', RiskEvidence()), coordinator.analyze('MSFT', RiskEvidence()))
        self.assertEqual(sum(call.args[0] == 'SPY' for call in fetch.call_args_list), 1)
        self.assertEqual([result.agents['market'] for result in results], ['completed', 'completed'])

    async def test_benchmark_cache_expires_and_failed_requests_retry(self):
        frame = history([100.] * 40)
        coordinator = Coordinator(AsyncMock(), None)
        with patch('adam.coordinator.fetch_history', return_value=frame) as fetch:
            self.assertIs(await coordinator.benchmark_history(), frame)
            self.assertIs(await coordinator.benchmark_history(), frame)
            self.assertEqual(fetch.call_count, 1)
            coordinator._benchmark_started -= 61
            self.assertIs(await coordinator.benchmark_history(), frame)
            self.assertEqual(fetch.call_count, 2)
        coordinator = Coordinator(AsyncMock(), None)
        with patch('adam.coordinator.fetch_history', side_effect=[RuntimeError('offline'), frame]) as fetch:
            with self.assertRaises(RuntimeError):
                await coordinator.benchmark_history()
            self.assertIs(await coordinator.benchmark_history(), frame)
            self.assertEqual(fetch.call_count, 2)

    async def test_actual_market_date_cannot_be_overridden_by_newer_user_timestamp(self):
        from datetime import datetime, timezone
        from test_risk import EVIDENCE
        current = datetime.now(timezone.utc)
        data = RiskEvidence(**(EVIDENCE | {'market_time': current}))
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            result = await Coordinator(model, LIMITS).analyze('AAPL', data)
        self.assertEqual(result.risk.status, 'fail')
        self.assertEqual(result.risk.action, 'NO_TRADE')
        self.assertIn('market_time', result.risk.reasons)
        self.assertLess(result.evaluated_risk_evidence.market_time, current)

    async def test_older_user_timestamp_and_missing_timestamp_are_preserved(self):
        from datetime import datetime, timezone
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        older = datetime(2020, 1, 1, tzinfo=timezone.utc)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)):
            coordinator = Coordinator(model, None)
            result = await coordinator.analyze('AAPL', RiskEvidence(market_time=older))
            missing = await coordinator.analyze('AAPL', RiskEvidence())
        self.assertEqual(result.evaluated_risk_evidence.market_time, older)
        self.assertIsNone(missing.evaluated_risk_evidence.market_time)

    async def test_news_prompt_is_compact_without_losing_saved_evidence(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        items = [{'title': 'AAPL ' + ('growth ' * 60), 'url': 'https://example.com/' + 'x' * 1000,
                  'published': '2026-10-01T00:00:00+00:00', 'source': 'Example',
                  'sentiment': {'label': 'positive', 'scores': {'positive': .8, 'neutral': .1, 'negative': .1}}}
                 for _ in range(20)]
        provider = AsyncMock(return_value={'basis': 'headlines', 'headlines': items, 'mean_scores': {'positive': .8, 'neutral': .1, 'negative': .1}})
        with patch('adam.coordinator.fetch_history', side_effect=RuntimeError('offline')):
            result = await Coordinator(model, None, news_provider=provider).analyze('AAPL', RiskEvidence())
        self.assertEqual(len(result.evidence['news']['headlines']), 20)
        prompt = next(call.args[2] for call in model.assess.await_args_list if call.args[1] == 'news')
        self.assertEqual(prompt['total_headlines'], 20)
        self.assertEqual(len(prompt['headlines']), 10)
        self.assertNotIn('url', prompt['headlines'][0])
        self.assertLessEqual(len(prompt['headlines'][0]['title']), 240)
