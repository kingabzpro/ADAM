import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd

from adam.study import (buy_and_hold_signals, decision_dates, directional_accuracy, ledger,
                        next_week_returns, run_weekly_study, sma_signals, summarize,
                        weekly_signal_series, window)
from test_decisions import decision


def make_frame(base, step):
    index = pd.bdate_range('2026-02-02', periods=80, tz='UTC')
    close = pd.Series([base + step * position for position in range(80)], index=index)
    return pd.DataFrame({'Open': close - 1, 'High': close + 1, 'Low': close - 2,
                         'Close': close, 'Volume': 1_000_000.}, index=index)


def fundamentals():
    return {'revenueGrowth': .1, 'mostRecentQuarter': datetime.now(timezone.utc).timestamp()}


class StudyTests(unittest.IsolatedAsyncioTestCase):
    async def test_weekly_study_records_signals_and_trades_next_open(self):
        frames = {'AAA': make_frame(100., 1.), 'BBB': make_frame(200., -1.), 'SPY': make_frame(500., 1.)}
        infos = {ticker: fundamentals() for ticker in ('AAA', 'BBB')}
        signals_map = {'AAA': 'BUY', 'BBB': 'SELL'}
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: (
            decision(agent, signals_map[ticker]).model_copy(update={'ticker': ticker}))
        study = await run_weekly_study(['AAA', 'BBB'], frames, infos, frames['SPY'],
                                       '2026-02-02', '2026-04-15', model=model)
        self.assertGreaterEqual(len(study['weeks']), 8)
        self.assertTrue(all(signal == 'BUY' for signal in study['signals']['AAA'].values()))
        self.assertTrue(all(signal == 'SELL' for signal in study['signals']['BBB'].values()))
        summary = summarize(['AAA', 'BBB'], frames, study)
        self.assertGreater(summary['portfolio']['strategy']['total_return'],
                           summary['portfolio']['buy_hold']['total_return'])
        self.assertGreater(summary['pooled']['accuracy']['BUY'], .99)
        self.assertGreater(summary['pooled']['accuracy']['SELL'], .99)
        trades = summary['per_ticker']['AAA']['strategy']['trades']
        self.assertEqual(trades[0]['side'], 'BUY')
        self.assertEqual(trades[0]['date'], '2026-02-09')
        self.assertEqual(summary['per_ticker']['BBB']['strategy']['trades'], [])

    async def test_missing_data_votes_no_trade(self):
        frames = {'AAA': make_frame(100., 1.), 'SPY': make_frame(500., 1.)}
        model = AsyncMock()
        model.assess.side_effect = lambda ticker, agent, evidence: (
            decision(agent, 'BUY').model_copy(update={'ticker': ticker}))
        # Starting late enough that AAA's window covers the 20-session benchmark comparison;
        # AAA still lacks fundamentals, so market and technical carry the vote.
        study = await run_weekly_study(['AAA', 'CCC'], frames, {}, frames['SPY'],
                                       '2026-03-02', '2026-03-15', model=model)
        self.assertTrue(all(study['signals']['AAA'][stamp] == 'BUY' for stamp in study['weeks']))
        self.assertTrue(all(signal == 'NO_TRADE' for signal in study['signals']['CCC'].values()))

    def test_decision_dates_are_last_session_of_each_week(self):
        index = pd.bdate_range('2026-02-02', periods=20, tz='UTC')
        weeks = decision_dates(index, '2026-02-02', '2026-02-27')
        self.assertEqual([stamp.date().isoformat() for stamp in weeks],
                         ['2026-02-06', '2026-02-13', '2026-02-20', '2026-02-27'])

    def test_ledger_executes_at_next_open_and_stays_flat_on_sell(self):
        frame = make_frame(100., 1.)
        weeks = decision_dates(frame.index, '2026-02-02', '2026-02-27')
        bought = ledger(frame, {weeks[0].date().isoformat(): 'BUY'})
        self.assertEqual(bought['trades'][0]['date'], '2026-02-09')
        self.assertGreater(bought['total_return'], 0)
        sold = ledger(frame, {weeks[0].date().isoformat(): 'SELL'})
        self.assertEqual(sold['trades'], [])
        self.assertEqual(sold['total_return'], 0)
        held = ledger(frame, buy_and_hold_signals(frame))
        self.assertEqual(held['trades'][0]['date'], frame.index[1].date().isoformat())

    def test_sma_rule_skips_warmup_and_buys_in_uptrend(self):
        frame = make_frame(100., 1.)
        weeks = decision_dates(frame.index, '2026-02-02', '2026-05-01')
        signals = sma_signals(frame, weeks)
        self.assertNotIn(weeks[0].date().isoformat(), signals)
        self.assertEqual(signals[weeks[-1].date().isoformat()], 'BUY')

    def test_weekly_signal_series_defaults_to_no_trade(self):
        frame = make_frame(100., 1.)
        series = weekly_signal_series(frame, {frame.index[2].date().isoformat(): 'BUY'})
        self.assertEqual(len(series), len(frame))
        self.assertEqual(series[2], 'BUY')
        self.assertEqual(series[0], 'NO_TRADE')

    def test_metrics_are_independent_of_bar_timestamps(self):
        # Real Yahoo daily bars carry intraday stamps; every date-based lookup must still hit.
        frame = make_frame(100., 1.)
        frame.index = frame.index + pd.Timedelta(hours=14, minutes=30)
        weeks = decision_dates(frame.index, '2026-02-02', '2026-03-31')
        sliced = window(frame, weeks[-1])
        self.assertEqual(sliced.index[-1].date().isoformat(), weeks[-1].date().isoformat())
        self.assertEqual(len(sliced), 40)
        signals = sma_signals(frame, weeks)
        self.assertEqual(signals[weeks[-1].date().isoformat()], 'BUY')
        returns = next_week_returns(frame, weeks)
        self.assertEqual(len(returns), 7)
        self.assertTrue(all(value > 0 for value in returns.values()))

    def test_directional_accuracy_counts_only_directed_signals(self):
        returns = {'2026-02-06': .01, '2026-02-13': -.02, '2026-02-20': .03, '2026-02-27': .05}
        result = directional_accuracy({'2026-02-06': 'BUY', '2026-02-13': 'BUY', '2026-02-20': 'SELL',
                                       '2026-02-27': 'HOLD'}, returns)
        self.assertEqual(result['accuracy']['BUY'], .5)
        self.assertEqual(result['accuracy']['SELL'], 0.0)
        self.assertEqual(result['counts'], {'BUY': 2, 'SELL': 1, 'HOLD': 1})
        self.assertAlmostEqual(result['mean_return']['BUY'], -.005)
        self.assertAlmostEqual(result['mean_return']['SELL'], .03)


if __name__ == '__main__':
    unittest.main()
