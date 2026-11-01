import unittest
from unittest.mock import AsyncMock, patch

from adam.coordinator import Coordinator
from adam.risk import RiskEvidence
from test_decisions import decision
from test_risk import LIMITS
from test_technical import history


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
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
