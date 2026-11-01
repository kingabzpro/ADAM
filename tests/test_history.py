import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from adam.coordinator import Coordinator
from adam.history import record_analysis, read_history
from adam.risk import RiskEvidence
from test_decisions import decision
from test_technical import history


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_and_risk_context_survive_reopening(self):
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: decision(agent)
        with patch('adam.coordinator.fetch_history', return_value=history([100.] * 40)), patch('adam.coordinator.fetch_fundamentals', side_effect=RuntimeError('unavailable')):
            result = await Coordinator(model, None).analyze('AAPL', RiskEvidence())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.sqlite'
            record_id = record_analysis(path, result, risk_evidence=RiskEvidence(), limits=None)
            records = read_history(path, ticker='AAPL')
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['id'], record_id)
            self.assertEqual(records[0]['ticker'], 'AAPL')
            self.assertEqual(records[0]['risk']['action'], 'NO_TRADE')
            self.assertEqual(records[0]['decisions']['market']['signal'], 'BUY')
            self.assertEqual(records[0]['evidence']['market']['last_close'], 100.0)
            self.assertEqual(records[0]['timeline'][-1]['stage'], 'risk')
            self.assertIsNone(records[0]['limits'])
            self.assertIn('evaluated_risk_evidence', records[0])
            self.assertEqual(read_history(path, ticker='MSFT'), [])

    async def test_invalid_limit_does_not_touch_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.sqlite'
            with self.assertRaises(ValueError):
                read_history(path, limit=0)
            self.assertFalse(path.exists())

    async def test_serialization_failure_does_not_create_database(self):
        from types import SimpleNamespace
        from adam.risk import RiskResult
        result = SimpleNamespace(ticker='AAPL', agents={}, decisions={}, evidence={}, errors={'invalid': float('nan')}, timeline=[], risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('missing',)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.sqlite'
            with self.assertRaises(ValueError):
                record_analysis(path, result, risk_evidence=RiskEvidence(), limits=None)
            self.assertFalse(path.exists())

    async def test_creates_missing_history_directory(self):
        from types import SimpleNamespace
        from adam.risk import RiskResult
        result = SimpleNamespace(ticker='AAPL', agents={}, decisions={}, evidence={}, errors={}, timeline=[], risk=RiskResult(status='insufficient_evidence', action='NO_TRADE', reasons=('missing',)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'history.sqlite'
            record_analysis(path, result, risk_evidence=RiskEvidence(), limits=None)
            self.assertEqual(read_history(path)[0]['ticker'], 'AAPL')
