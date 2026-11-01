import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from adam.history import record_analysis
from adam.paper import create_account, execute_paper, paper_status, paper_trades
from adam.risk import RiskEvidence, RiskResult
from test_decisions import decision
from test_risk import EVIDENCE, LIMITS


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'history.sqlite'
        self.now = datetime.now(timezone.utc)
        self.evidence = RiskEvidence(**(EVIDENCE | {'market_time': self.now}))
        create_account(self.path, 'AAPL', initial_cash=10000, cost_bps=100)

    def record(self, signal='BUY', status='pass', evidence=None):
        data = evidence or self.evidence
        decisions = {agent: decision(agent, signal) for agent in ('market', 'forecast')}
        market = {'as_of': self.now.isoformat(), 'annualized_volatility': .2, 'last_close': 100,
                  'last_volume': 20000, 'comparison_status': 'available'}
        forecast = {'lower_returns': [-.1]}
        result = SimpleNamespace(ticker='AAPL', agents={}, evidence={'market': market, 'forecast': forecast},
                                 decisions=decisions, risk=RiskResult(status=status, action=signal if status in ('pass', 'warning') else 'NO_TRADE', reasons=()),
                                 errors={}, timeline=[], evaluated_risk_evidence=data)
        return record_analysis(self.path, result, risk_evidence=data, limits=LIMITS)

    def test_buy_sell_accounting_and_duplicate_prevention(self):
        buy_id = self.record('BUY')
        buy = execute_paper(self.path, buy_id, price=100)
        self.assertAlmostEqual(buy['shares'], 500 / 101)
        self.assertAlmostEqual(buy['fee'], 500 / 101)
        self.assertAlmostEqual(paper_status(self.path, 'AAPL', price=100)['cash'], 9500)
        with self.assertRaisesRegex(ValueError, 'already'):
            execute_paper(self.path, buy_id, price=100)
        sell_id = self.record('SELL')
        sell = execute_paper(self.path, sell_id, price=120)
        self.assertEqual(sell['side'], 'SELL')
        status = paper_status(self.path, 'AAPL', price=120)
        self.assertEqual(status['shares'], 0)
        self.assertAlmostEqual(status['cash'], 9500 + (500 / 101) * 120 * .99)
        self.assertEqual(status['trade_count'], 2)
        trades = paper_trades(self.path, 'AAPL')
        self.assertEqual([trade['side'] for trade in trades], ['BUY', 'SELL'])
        self.assertEqual(trades[0]['risk_check']['result']['action'], 'BUY')

    def test_rejects_no_trade_stale_analysis_and_invalid_prices(self):
        denied_id = self.record(status='insufficient_evidence')
        with self.assertRaises(ValueError):
            execute_paper(self.path, denied_id, price=100)
        buy_id = self.record()
        with self.assertRaises(ValueError):
            execute_paper(self.path, buy_id, price=100, now=self.now + timedelta(hours=2))
        for value in (0, -1, float('nan'), True):
            with self.subTest(price=value), self.assertRaises(ValueError):
                execute_paper(self.path, buy_id, price=value)
        self.assertEqual(paper_status(self.path, 'AAPL')['trade_count'], 0)

    def test_duplicate_account_does_not_overwrite_balance(self):
        with self.assertRaises(ValueError):
            create_account(self.path, 'AAPL', initial_cash=1)
        self.assertEqual(paper_status(self.path, 'AAPL')['cash'], 10000)

    def test_underflowing_position_is_not_recorded(self):
        import sqlite3
        from contextlib import closing
        analysis_id = self.record()
        with closing(sqlite3.connect(self.path)) as database, database:
            database.execute('UPDATE paper_accounts SET cash = 1e-300 WHERE ticker = ?', ('AAPL',))
        with self.assertRaises(ValueError):
            execute_paper(self.path, analysis_id, price=1e300)
        self.assertEqual(paper_status(self.path, 'AAPL')['trade_count'], 0)

    def test_missing_portfolio_evidence_cannot_be_filled_by_simulator(self):
        incomplete = self.evidence.model_copy(update={'concentration': None})
        analysis_id = self.record(evidence=incomplete)
        with self.assertRaisesRegex(ValueError, 'concentration'):
            execute_paper(self.path, analysis_id, price=100)
        self.assertEqual(paper_status(self.path, 'AAPL')['trade_count'], 0)

    def test_expected_ticker_mismatch_never_records_a_trade(self):
        analysis_id = self.record()
        with self.assertRaisesRegex(ValueError, 'ticker'):
            execute_paper(self.path, analysis_id, price=100, expected_ticker='MSFT')
        self.assertEqual(paper_status(self.path, 'AAPL')['trade_count'], 0)
        self.assertEqual(execute_paper(self.path, analysis_id, price=100, expected_ticker='aapl')['side'], 'BUY')

    def test_missing_forecast_evidence_cannot_be_replaced_by_user_downside(self):
        import json
        import sqlite3
        from contextlib import closing
        analysis_id = self.record()
        with closing(sqlite3.connect(self.path)) as database, database:
            raw = database.execute('SELECT payload FROM analysis_history WHERE id = ?', (analysis_id,)).fetchone()[0]
            payload = json.loads(raw)
            payload['evidence']['forecast'] = None
            database.execute('UPDATE analysis_history SET payload = ? WHERE id = ?', (json.dumps(payload), analysis_id))
        with self.assertRaises(ValueError):
            execute_paper(self.path, analysis_id, price=100)
        self.assertEqual(paper_status(self.path, 'AAPL')['trade_count'], 0)

    def test_simultaneous_execution_cannot_duplicate_a_trade(self):
        from concurrent.futures import ThreadPoolExecutor
        analysis_id = self.record()
        def attempt():
            try:
                return execute_paper(self.path, analysis_id, price=100)['side']
            except ValueError:
                return 'rejected'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertCountEqual(results, ['BUY', 'rejected'])
        self.assertEqual(paper_status(self.path, 'AAPL')['trade_count'], 1)

    def test_command_reports_status(self):
        import json
        from unittest.mock import patch
        from adam.paper import main
        with patch('sys.argv', ['adam-paper', '--db', str(self.path), 'status', 'AAPL']), \
             patch('sys.stdout', new_callable=__import__('io').StringIO) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())['cash'], 10000)

    def test_existing_database_without_paper_tables_has_clear_error(self):
        import sqlite3
        from contextlib import closing
        empty = Path(self.directory.name) / 'empty.sqlite'
        with closing(sqlite3.connect(empty)):
            pass
        with self.assertRaisesRegex(ValueError, 'account'):
            paper_status(empty, 'AAPL')
        with self.assertRaisesRegex(ValueError, 'analysis'):
            execute_paper(empty, 1, price=100)
