import unittest

import pandas as pd

from adam.backtest import backtest


class BacktestTests(unittest.TestCase):
    def prices(self):
        return pd.DataFrame({'Open': [10., 10., 12., 11.], 'Close': [10., 12., 11., 11.]},
                            index=pd.date_range('2026-01-01', periods=4, tz='UTC'))

    def test_signals_execute_next_session_and_charge_costs(self):
        result = backtest(self.prices(), ['BUY', 'SELL', 'HOLD', 'BUY'], initial_cash=100, cost_bps=100)
        self.assertEqual([trade['side'] for trade in result['trades']], ['BUY', 'SELL'])
        self.assertEqual(result['trades'][0]['date'], '2026-01-02')
        self.assertEqual(result['trades'][1]['date'], '2026-01-03')
        self.assertAlmostEqual(result['ending_equity'], 100 / 1.01 * 1.2 * .99)
        self.assertGreater(result['transaction_costs'], 0)
        self.assertAlmostEqual(result['total_return'], result['ending_equity'] / 100 - 1)

    def test_no_trade_keeps_position_and_last_signal_is_not_executed(self):
        result = backtest(self.prices(), ['BUY', 'NO_TRADE', 'NO_TRADE', 'SELL'], initial_cash=100)
        self.assertEqual(len(result['trades']), 1)
        self.assertAlmostEqual(result['ending_equity'], 110)
        self.assertGreaterEqual(result['max_drawdown'], 0)

    def test_bad_inputs_rejected(self):
        for frame, signals, kwargs in [
            (self.prices().iloc[::-1], ['BUY'] * 4, {}),
            (self.prices().assign(Open=[10, 0, 12, 11]), ['BUY'] * 4, {}),
            (self.prices(), ['BUY'], {}),
            (self.prices(), ['BUY', 'BUY', 'BUY', 'MAYBE'], {}),
            (self.prices(), ['BUY'] * 4, {'cost_bps': 10000}),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backtest(frame, signals, **kwargs)
