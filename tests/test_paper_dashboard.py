import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nicegui import ui

from adam.config import Config
from adam.dashboard import dashboard
from adam.paper import paper_status


class PaperDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def click(self, button):
        before = asyncio.all_tasks()
        with patch('nicegui.core.loop', asyncio.get_running_loop()), \
             patch('nicegui.core.is_loop_running', return_value=True):
            next(iter(button._event_listeners.values())).handler(None)
            created = asyncio.all_tasks() - before
            self.assertTrue(created)
            await asyncio.wait_for(asyncio.gather(*created), timeout=5)

    async def test_dashboard_creates_and_reads_a_local_paper_account(self):
        with tempfile.TemporaryDirectory() as directory, patch('adam.dashboard.Path.home', return_value=Path(directory)):
            with ui.column() as root:
                dashboard(Config())
            try:
                elements = list(ui.context.client.elements.values())
                button = next(item for item in elements if getattr(item, 'text', None) == 'Create paper account')
                await self.click(button)
                path = Path(directory) / '.adam' / 'history.sqlite'
                account = paper_status(path, 'AAPL')
                self.assertEqual(account['cash'], 10000)
                self.assertEqual(account['trade_count'], 0)
                from datetime import datetime, timezone
                from types import SimpleNamespace
                from adam.history import record_analysis
                from adam.risk import RiskEvidence, RiskResult
                from test_decisions import decision
                from test_risk import EVIDENCE, LIMITS
                now = datetime.now(timezone.utc)
                risk = RiskEvidence(**(EVIDENCE | {'market_time': now}))
                result = SimpleNamespace(ticker='AAPL', agents={}, decisions={name: decision(name) for name in ('market', 'forecast')},
                    evidence={'market': {'as_of': now.isoformat(), 'annualized_volatility': .2, 'last_close': 100,
                                         'last_volume': 20000, 'comparison_status': 'available'},
                              'forecast': {'lower_returns': [-.1]}},
                    risk=RiskResult(status='pass', action='BUY', reasons=()), errors={}, timeline=[], evaluated_risk_evidence=risk)
                identifier = record_analysis(path, result, risk_evidence=risk, limits=LIMITS)
                ticker = next(e for e in elements if e._props.get('label') == 'Paper account ticker')
                analysis = next(e for e in elements if e._props.get('label') == 'Saved analysis ID')
                price = next(e for e in elements if e._props.get('label') == 'Hypothetical price')
                button = next(e for e in elements if getattr(e, 'text', None) == 'Record simulated trade')
                offline = next(e for e in elements if getattr(e, 'text', None) == 'Offline: use saved snapshots, always No Trade')
                ticker.value, analysis.value, price.value = 'MSFT', identifier, 100
                await self.click(button)
                self.assertEqual(paper_status(path, 'AAPL')['trade_count'], 0)
                ticker.value = 'AAPL'
                offline.value = True
                await self.click(button)
                self.assertEqual(paper_status(path, 'AAPL')['trade_count'], 0)
                offline.value = False
                await self.click(button)
                self.assertEqual(paper_status(path, 'AAPL')['trade_count'], 1)
            finally:
                root.delete()
