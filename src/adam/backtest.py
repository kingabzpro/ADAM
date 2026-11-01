from math import isfinite, sqrt

import numpy as np
import pandas as pd


def backtest(prices, signals, *, initial_cash=10000.0, cost_bps=0.0, position_fractions=None):
    """Simulate long/cash trades at the open after each close-time signal.

    Inputs must be point-in-time, consistently adjusted prices and risk-gated signals.
    A position fraction is the fraction of available cash committed on entry,
    including fees. Positions are marked to the final close, not liquidated.
    """
    if (len(prices) < 2 or not {'Open', 'Close'}.issubset(prices.columns)
            or not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None
            or not prices.index.is_monotonic_increasing or prices.index.has_duplicates
            or len(signals) != len(prices) or any(s not in {'BUY', 'SELL', 'HOLD', 'NO_TRADE'} for s in signals)):
        raise ValueError('Ordered dated prices and one valid signal per session are required')
    if (isinstance(initial_cash, bool) or isinstance(cost_bps, bool)
            or not isinstance(initial_cash, (int, float)) or not isinstance(cost_bps, (int, float))
            or not isfinite(initial_cash) or not isfinite(cost_bps)
            or initial_cash <= 0 or not 0 <= cost_bps < 10000):
        raise ValueError('Invalid initial cash or transaction cost')
    if position_fractions is None:
        position_fractions = [1.0] * len(prices)
    if (len(position_fractions) != len(prices) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not 0 <= value <= 1 for value in position_fractions)):
        raise ValueError('A valid position fraction is required for every session')
    try:
        data = prices[['Open', 'Close']].apply(pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError('Prices must be numeric') from error
    if not np.isfinite(data.to_numpy()).all() or (data <= 0).to_numpy().any():
        raise ValueError('Prices must be positive and finite')

    cash, shares, total_costs = float(initial_cash), 0.0, 0.0
    purchase_cost = 0.0
    closed_profits, trades = [], []
    equity = [cash]
    rate = cost_bps / 10000
    for i in range(1, len(data)):
        signal = signals[i - 1]
        open_price = float(data['Open'].iloc[i])
        date = data.index[i].date().isoformat()
        allocation = cash * position_fractions[i - 1]
        if signal == 'BUY' and shares == 0 and allocation > 0:
            shares = allocation / (open_price * (1 + rate))
            fee = shares * open_price * rate
            purchase_cost = allocation
            cash -= allocation
            total_costs += fee
            trades.append({'date': date, 'side': 'BUY', 'price': open_price, 'shares': shares, 'fee': fee})
        elif signal == 'SELL' and shares > 0:
            proceeds = shares * open_price
            fee = proceeds * rate
            cash += proceeds - fee
            total_costs += fee
            closed_profits.append(proceeds - fee - purchase_cost)
            trades.append({'date': date, 'side': 'SELL', 'price': open_price, 'shares': shares, 'fee': fee})
            shares = 0.0
        equity.append(cash + shares * float(data['Close'].iloc[i]))

    equity_array = np.asarray(equity)
    daily_returns = equity_array[1:] / equity_array[:-1] - 1
    volatility = daily_returns.std(ddof=1) if len(daily_returns) > 1 else 0.0
    gross_profit = sum(p for p in closed_profits if p > 0)
    gross_loss = -sum(p for p in closed_profits if p < 0)
    return {
        'ending_equity': float(equity_array[-1]), 'total_return': float(equity_array[-1] / initial_cash - 1),
        'max_drawdown': float(np.max(1 - equity_array / np.maximum.accumulate(equity_array))),
        'sharpe': float(sqrt(252) * daily_returns.mean() / volatility) if volatility > 0 else None,
        'win_rate': sum(p > 0 for p in closed_profits) / len(closed_profits) if closed_profits else None,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else None,
        'transaction_costs': float(total_costs), 'trades': trades,
        'equity': [{'date': date.date().isoformat(), 'value': float(value)} for date, value in zip(data.index, equity, strict=True)],
    }
