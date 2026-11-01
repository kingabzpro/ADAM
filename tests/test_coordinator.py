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
