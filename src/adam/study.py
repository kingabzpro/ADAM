"""Weekly historical study: run the decision pipeline on point-in-time windows.

The study slices each ticker's daily history at every weekly decision date, runs the
multi-agent consensus on that window, and maps the resulting weekly signals through the
long/cash ledger from adam.backtest (a signal decided at a week's close executes at the
next session's open). Fundamentals are today's data and no news agent runs, so results
are research measurements with known lookahead, not performance guarantees.
"""
import argparse
import asyncio
import json
import sys
from math import sqrt
from unittest.mock import patch

import numpy as np
import pandas as pd

TOP_TICKERS = ('AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'TSLA', 'LLY', 'AVGO',
               'JPM', 'V', 'XOM', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'HD', 'WMT',
               'ORCL', 'MRK', 'ABBV', 'CVX', 'CRM')

WINDOW_SESSIONS = 63
SMA_SESSIONS = 20


def decision_dates(calendar, start, end):
    """Return the last session of each ISO week whose date falls in [start, end]."""
    if not isinstance(calendar, pd.DatetimeIndex) or calendar.tz is None or calendar.has_duplicates:
        raise ValueError('A tz-aware session calendar without duplicates is required')
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    iso = calendar.isocalendar()
    last_of_week = {}
    for stamp, key in zip(calendar, zip(iso.year, iso.week), strict=True):
        last_of_week[key] = stamp
    return sorted(stamp for stamp in last_of_week.values() if start <= stamp.date() <= end)


def window(frame, week_end):
    """Return the point-in-time price window ending at week_end (inclusive), matched by date."""
    key = pd.Timestamp(week_end).date()
    return frame[[stamp.date() <= key for stamp in frame.index]].tail(WINDOW_SESSIONS)


def signal_at_week(result):
    """Consensus signal for one analysis; NO_TRADE when too few agents vote."""
    from adam.presentation import vote_summary
    signal = vote_summary(list(result.decisions.values()))['signal'] if result.decisions else None
    return signal or 'NO_TRADE'


async def run_weekly_study(tickers, frames, fundamentals, spy, start, end, *,
                           model, forecaster=None, concurrency=4, inference_concurrency=2):
    """Run the pipeline at every weekly decision date and return signals, errors, and dates."""
    from adam.coordinator import Coordinator
    from adam.risk import RiskEvidence

    calendar = spy.index
    weeks = decision_dates(calendar, start, end)
    if len(weeks) < 2:
        raise ValueError('At least two weekly decision dates are required')
    coordinator = Coordinator(model, None, forecaster=forecaster,
                              inference_concurrency=inference_concurrency, data_concurrency=concurrency)
    slots = asyncio.Semaphore(concurrency)
    signals = {ticker: {} for ticker in tickers}
    decisions = {ticker: {} for ticker in tickers}
    errors = {ticker: {} for ticker in tickers}

    async def analyze(ticker, week_end):
        async with slots:
            return await coordinator.analyze(ticker, RiskEvidence())

    for index, week_end in enumerate(weeks):
        with patch('adam.coordinator.fetch_history',
                   side_effect=lambda ticker, period='3mo': window(frames[ticker], week_end)), \
             patch('adam.coordinator.fetch_fundamentals', side_effect=lambda ticker: fundamentals[ticker]):
            outcomes = await asyncio.gather(*(analyze(ticker, week_end) for ticker in tickers),
                                            return_exceptions=True)
        for ticker, outcome in zip(tickers, outcomes, strict=True):
            stamp = week_end.date().isoformat()
            if isinstance(outcome, Exception):
                errors[ticker][stamp] = str(outcome)
                signals[ticker][stamp] = 'NO_TRADE'
            else:
                signals[ticker][stamp] = signal_at_week(outcome)
                decisions[ticker][stamp] = list(outcome.decisions.values())
        voted = sum(1 for ticker in tickers if signals[ticker][week_end.date().isoformat()] != 'NO_TRADE')
        print(f'week {index + 1}/{len(weeks)} ({week_end.date()}): {voted}/{len(tickers)} votes', flush=True)
    return {'weeks': [stamp.date().isoformat() for stamp in weeks], 'signals': signals,
            'decisions': decisions, 'errors': errors}


def signals_for_margin(decisions_by_week, margin=0.0):
    """Recompute each week's consensus signal at a directional margin."""
    from adam.decisions import consensus
    return {stamp: (consensus(list(week_decisions), margin=margin).signal or 'NO_TRADE')
            for stamp, week_decisions in decisions_by_week.items()}


def margin_sweep(decisions, tickers, frames, study, margins, *, cost_bps=0.0):
    """Aggregate strategy behavior for each margin from one pipeline pass."""
    rows = []
    for margin in margins:
        signals = {ticker: signals_for_margin(decisions[ticker], margin) for ticker in tickers}
        summary = summarize(tickers, frames, {'weeks': study['weeks'], 'signals': signals,
                                              'errors': study['errors']}, cost_bps=cost_bps)
        counts = summary['pooled']['counts']
        rows.append({'margin': margin,
                     'strategy_return': summary['portfolio']['strategy']['total_return'],
                     'buy_hold_return': summary['portfolio']['buy_hold']['total_return'],
                     'buy_accuracy': summary['pooled']['accuracy']['BUY'],
                     'sell_accuracy': summary['pooled']['accuracy']['SELL'],
                     'buy_weeks': counts['BUY'], 'sell_weeks': counts['SELL'], 'hold_weeks': counts['HOLD']})
    return rows


def weekly_signal_series(frame, signals_by_date):
    """Expand weekly decisions onto daily rows; NO_TRADE elsewhere, so each decision
    executes at the next session's open (the following week's first open)."""
    return [signals_by_date.get(stamp.date().isoformat(), 'NO_TRADE') for stamp in frame.index]


def ledger(frame, signals_by_date, *, cost_bps=0.0):
    """Run the long/cash ledger for one ticker's daily frame and weekly signals."""
    from adam.backtest import backtest
    return backtest(frame, weekly_signal_series(frame, signals_by_date), cost_bps=cost_bps)


def buy_and_hold_signals(frame):
    return {frame.index[0].date().isoformat(): 'BUY'}


def sma_signals(frame, weeks, window=SMA_SESSIONS):
    """Trailing moving-average rule sampled at the weekly decision dates (date-matched)."""
    close = frame['Close'].copy()
    close.index = [stamp.date() for stamp in close.index]
    signals = {}
    for stamp in weeks:
        history = close.loc[:stamp.date()].tail(window + 1)
        if len(history) <= window:
            continue
        average = float(history.iloc[:-1].mean())
        last = float(history.iloc[-1])
        signals[stamp.date().isoformat()] = 'BUY' if last > average else 'SELL' if last < average else 'HOLD'
    return signals


def next_week_returns(frame, weeks):
    """Close-to-close return from each decision date to the next one (date-matched)."""
    close_by_date = {stamp.date(): float(value) for stamp, value in frame['Close'].items()}
    returns = {}
    for index in range(len(weeks) - 1):
        current, following = weeks[index].date(), weeks[index + 1].date()
        if current in close_by_date and following in close_by_date:
            returns[current.isoformat()] = close_by_date[following] / close_by_date[current] - 1
    return returns


def directional_accuracy(signals_by_date, returns_by_date):
    """BUY calls count as correct when the next weekly return is positive; SELL the reverse."""
    correct = {'BUY': 0, 'SELL': 0}
    total = {'BUY': 0, 'SELL': 0}
    weekly = {'BUY': [], 'SELL': [], 'HOLD': []}
    for stamp, signal in signals_by_date.items():
        weekly_return = returns_by_date.get(stamp)
        if weekly_return is None or signal not in weekly:
            continue
        weekly[signal].append(weekly_return)
        if signal in correct:
            is_correct = weekly_return > 0 if signal == 'BUY' else weekly_return < 0
            correct[signal] += is_correct
            total[signal] += 1
    return {'accuracy': {signal: (correct[signal] / total[signal] if total[signal] else None)
                         for signal in correct},
            'mean_return': {signal: (sum(values) / len(values) if values else None)
                            for signal, values in weekly.items()},
            'counts': {signal: len(values) for signal, values in weekly.items()},
            'correct': dict(correct),
            'weekly_returns': weekly}


def portfolio_curve(ledgers, initial_cash=10000.0):
    """Equal-weight mean of per-ticker normalized equity curves sharing one calendar."""
    curves = [pd.Series([point['value'] / initial_cash for point in ledger['equity']],
                        index=pd.DatetimeIndex([point['date'] for point in ledger['equity']]))
              for ledger in ledgers.values()]
    if not curves:
        raise ValueError('At least one ledger is required')
    aligned = pd.concat(curves, axis=1, join='inner')
    if aligned.empty:
        raise ValueError('Ledger calendars do not overlap')
    mean_curve = aligned.mean(axis=1)
    daily = mean_curve.pct_change().dropna()
    volatility = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    return {'total_return': float(mean_curve.iloc[-1] - 1),
            'max_drawdown': float(np.max(1 - mean_curve / mean_curve.cummax())),
            'sharpe': float(sqrt(252) * daily.mean() / volatility) if volatility > 0 else None,
            'weeks': [stamp.date().isoformat() for stamp in aligned.index]}


def summarize(tickers, frames, study, *, cost_bps=0.0, window=SMA_SESSIONS):
    """Strategy, buy-and-hold, and SMA-rule ledgers plus signal quality for every ticker."""
    weeks = [pd.Timestamp(stamp, tz='UTC') for stamp in study['weeks']]
    per_ticker = {}
    for ticker in tickers:
        frame, signals = frames[ticker], study['signals'][ticker]
        returns = next_week_returns(frame, weeks)
        per_ticker[ticker] = {
            'strategy': ledger(frame, signals, cost_bps=cost_bps),
            'buy_hold': ledger(frame, buy_and_hold_signals(frame), cost_bps=cost_bps),
            'sma_rule': ledger(frame, sma_signals(frame, weeks, window), cost_bps=cost_bps),
            'signals': directional_accuracy(signals, returns),
        }
    portfolio = {name: portfolio_curve({ticker: per_ticker[ticker][name] for ticker in tickers})
                 for name in ('strategy', 'buy_hold', 'sma_rule')}
    pooled = {'accuracy': {}, 'mean_return': {}, 'counts': {}}
    for signal in ('BUY', 'SELL', 'HOLD'):
        if signal == 'HOLD':
            pooled['counts'][signal] = sum(per_ticker[ticker]['signals']['counts'][signal] for ticker in tickers)
            pooled['mean_return'][signal] = None
            pooled['accuracy'][signal] = None
            continue
        correct = sum(per_ticker[ticker]['signals']['correct'][signal] for ticker in tickers)
        total = sum(per_ticker[ticker]['signals']['counts'][signal] for ticker in tickers)
        values = [value for ticker in tickers
                  for value in per_ticker[ticker]['signals']['weekly_returns'][signal]]
        pooled['accuracy'][signal] = correct / total if total else None
        pooled['mean_return'][signal] = sum(values) / len(values) if values else None
        pooled['counts'][signal] = total
    return {'per_ticker': per_ticker, 'portfolio': portfolio, 'pooled': pooled,
            'errors': study['errors'], 'weeks': study['weeks']}


def markdown_report(summary, *, start, end, tickers, cost_bps):
    def pct(value):
        return 'n/a' if value is None else f'{value * 100:+.1f}%'

    def ratio(value):
        return 'n/a' if value is None else f'{value:.2f}'

    lines = [f'# ADAM weekly backtest: {start} to {end}',
             '',
             f'Universe: {len(tickers)} tickers ({", ".join(tickers)}). Weekly decisions from the '
             f'four-agent consensus (market, technical, forecast, fundamentals; no historical news), '
             f'executed long/flat at the next session\'s open, transaction cost {cost_bps} bps per trade.',
             '',
             '## Aggregate (equal weight)',
             '',
             '| Strategy | Total return | Sharpe | Max drawdown |',
             '|---|---|---|---|']
    for name, label in (('strategy', 'ADAM weekly consensus'), ('buy_hold', 'Buy and hold'),
                        ('sma_rule', 'SMA-20 rule')):
        curve = summary['portfolio'][name]
        lines.append(f"| {label} | {pct(curve['total_return'])} | {ratio(curve['sharpe'])} "
                     f"| {pct(curve['max_drawdown'])} |")
    pooled = summary['pooled']
    lines += ['', '## Signal quality (pooled over all ticker-weeks)', '',
              '| Signal | Weeks | Mean next-week return | Directional accuracy |', '|---|---|---|---|']
    for signal in ('BUY', 'SELL', 'HOLD'):
        lines.append(f"| {signal} | {pooled['counts'][signal]} | {pct(pooled['mean_return'][signal])} "
                     f"| {pct(pooled['accuracy'][signal])} |")
    lines += ['', '## Per ticker', '',
              '| Ticker | Strategy | Buy and hold | SMA rule | BUY accuracy | SELL accuracy | Errors |',
              '|---|---|---|---|---|---|---|']
    for ticker in tickers:
        data = summary['per_ticker'][ticker]
        accuracy = data['signals']['accuracy']
        error_count = len(summary['errors'].get(ticker, {}))
        lines.append(f"| {ticker} | {pct(data['strategy']['total_return'])} "
                     f"| {pct(data['buy_hold']['total_return'])} | {pct(data['sma_rule']['total_return'])} "
                     f"| {pct(accuracy['BUY'])} | {pct(accuracy['SELL'])} | {error_count} |")
    sweep = summary.get('sweep')
    if sweep:
        lines += ['', '## Margin sweep (one pipeline pass, consensus recomputed per margin)', '',
                  '| Margin | Strategy return | BUY weeks | SELL weeks | HOLD weeks | BUY accuracy | SELL accuracy |',
                  '|---|---|---|---|---|---|---|']
        for row in sweep:
            lines.append(f"| {row['margin']:g} | {pct(row['strategy_return'])} | {row['buy_weeks']} "
                         f"| {row['sell_weeks']} | {row['hold_weeks']} | {pct(row['buy_accuracy'])} "
                         f"| {pct(row['sell_accuracy'])} |")
    lines += ['', '## Caveats', '',
              '- Fundamentals are today\'s data applied to historical weeks (Yahoo has no point-in-time fundamentals), '
              'so the fundamentals agent carries lookahead bias.',
              '- No historical news is replayed; votes come from four agents.',
              '- The universe is today\'s largest companies, which biases the study toward survivors.',
              '- Weekly closes and opens come from Yahoo-adjusted daily bars; signals decide at a week\'s last '
              'session and execute at the next session\'s open.',
              '- Decision quality does not establish profitability; this is a research measurement.',
              '']
    return '\n'.join(lines)


def fetch_study_data(tickers, start, end):
    """Fetch each ticker and SPY once with lookback, plus current fundamentals."""
    from adam.fundamentals import fetch_fundamentals
    from adam.market import fetch_history_between
    lookback_start = pd.Timestamp(start, tz='UTC') - pd.Timedelta(weeks=18)
    fetch_end = pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=14)
    frames, fundamentals = {}, {}
    symbols = [*tickers, 'SPY']
    for index, ticker in enumerate(symbols, start=1):
        frame = fetch_history_between(ticker, lookback_start.date(), fetch_end.date())
        if frame.empty:
            raise ValueError(f'No price history returned for {ticker}')
        frames[ticker] = frame
        print(f'data {index}/{len(symbols)}: {ticker} ({len(frame)} sessions)', flush=True)
    for index, ticker in enumerate(tickers, start=1):
        fundamentals[ticker] = fetch_fundamentals(ticker)
        print(f'fundamentals {index}/{len(tickers)}: {ticker}', flush=True)
    return frames, fundamentals, frames['SPY']


async def run_study(tickers, start, end, *, concurrency=4, cost_bps=5.0, margin=0.0):
    import httpx
    from adam.d3 import D3Client
    from adam.forecast import shared_forecaster

    frames, fundamentals, spy = fetch_study_data(tickers, start, end)
    async with httpx.AsyncClient(timeout=180, trust_env=False) as http:
        study = await run_weekly_study(tickers, frames, fundamentals, spy, start, end,
                                       model=D3Client(http),
                                       forecaster=await asyncio.to_thread(shared_forecaster),
                                       concurrency=concurrency)
    summary = summarize(tickers, frames, study, cost_bps=cost_bps)
    summary['sweep'] = margin_sweep(study['decisions'], tickers, frames, study,
                                    (0.0, .025, .05, .075, .1, .15, .2), cost_bps=cost_bps)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Weekly historical backtest of ADAM consensus decisions.')
    parser.add_argument('--start', default='2026-01-01')
    parser.add_argument('--end', default='2026-06-30')
    parser.add_argument('--tickers', default=','.join(TOP_TICKERS))
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--cost-bps', type=float, default=5.0)
    parser.add_argument('--margin', type=float, default=0.075,
                        help='Minimum directional probability lean before acting (0 is fully decisive)')
    parser.add_argument('--out', help='Write the markdown report to this path')
    args = parser.parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(',') if ticker.strip()]
    if not tickers:
        print('No tickers given', file=sys.stderr)
        raise SystemExit(1)
    try:
        summary = asyncio.run(run_study(tickers, args.start, args.end, concurrency=args.concurrency,
                                        cost_bps=args.cost_bps, margin=args.margin))
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Backtest failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({'tickers': tickers, 'weeks': len(summary['weeks']),
                      'strategy_return': summary['portfolio']['strategy']['total_return'],
                      'buy_hold_return': summary['portfolio']['buy_hold']['total_return'],
                      'sma_rule_return': summary['portfolio']['sma_rule']['total_return'],
                      'pooled': summary['pooled'], 'margin_sweep': summary['sweep']}))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as file:
            file.write(markdown_report(summary, start=args.start, end=args.end,
                                       tickers=tickers, cost_bps=args.cost_bps))
        print(f'report written to {args.out}')


if __name__ == '__main__':
    main()
