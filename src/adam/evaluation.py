import numpy as np
import pandas as pd

from adam.forecast import forecast_evidence


def compare_rule_baseline(prices, decisions, evidence, limits, *, window=20, initial_cash=10000.0, cost_bps=0.0):
    """Compare consensus to a trailing moving-average rule on the same sessions.

    Inputs must be point-in-time; historical provenance is the caller's
    responsibility. Both methods use the same risk evidence and execution model.
    """
    from adam.backtest import backtest
    from adam.decisions import Consensus
    from adam.risk import evaluate_risk

    if type(window) is not int or not 2 <= window <= len(prices):
        raise ValueError('Window must cover at least two available sessions')
    model = compare_voting(prices, decisions, evidence, limits,
                           weights={agent: 1 for agent in ('market', 'forecast', 'news', 'technical', 'fundamentals')},
                           initial_cash=initial_cash, cost_bps=cost_bps)['majority']
    closes = pd.to_numeric(prices['Close'], errors='raise').astype(float)
    averages = closes.rolling(window).mean()
    signals = []
    for i, (when, inputs) in enumerate(zip(prices.index, evidence, strict=True)):
        if i < window - 1:
            signals.append('NO_TRADE')
            model['signals'][i] = 'NO_TRADE'
            continue
        signal = 'BUY' if closes.iloc[i] > averages.iloc[i] else 'SELL' if closes.iloc[i] < averages.iloc[i] else 'HOLD'
        vote = Consensus(signal=signal, counts={signal: 1}, disagreement=False, abstained=())
        signals.append(evaluate_risk(vote, inputs, limits, now=when.to_pydatetime()).action)
    fractions = [item.position_fraction or 0 for item in evidence]
    options = {'initial_cash': initial_cash, 'cost_bps': cost_bps, 'position_fractions': fractions}
    model['backtest'] = backtest(prices, model['signals'], **options)
    return {'model': model, 'rule': {'signals': signals, 'backtest': backtest(prices, signals, **options)}}


def compare_voting(prices, decisions, evidence, limits, *, weights, initial_cash=10000.0, cost_bps=0.0):
    """Compare precomputed point-in-time votes; timestamps denote signal times.

    Callers must use weights fixed before this evaluation and evidence available at
    each timestamp. This function cannot verify those provenance requirements.
    """
    from adam.backtest import backtest
    from adam.decisions import consensus
    from adam.risk import evaluate_risk

    if (not isinstance(weights, dict) or not weights
            or not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None
            or len(decisions) != len(prices) or len(evidence) != len(prices)):
        raise ValueError('Each dated session needs decisions and risk evidence')
    tickers = {item.ticker for session in decisions for item in session}
    if len(tickers) > 1:
        raise ValueError('All sessions must belong to one ticker')
    results = {}
    for name, vote_weights in (('majority', None), ('weighted', weights)):
        signals = []
        for when, session, inputs in zip(prices.index, decisions, evidence, strict=True):
            vote = consensus(session, weights=vote_weights)
            risk = evaluate_risk(vote, inputs, limits, now=when.to_pydatetime())
            available = {d.agent for d in session if d.status == 'completed' and d.reliable and not d.stale}
            signals.append(risk.action if {'market', 'forecast'} <= available else 'NO_TRADE')
        results[name] = {'signals': signals, 'backtest': backtest(
            prices, signals, initial_cash=initial_cash, cost_bps=cost_bps,
            position_fractions=[item.position_fraction or 0 for item in evidence])}
    return results


def evaluate_forecast(prices, forecaster, *, initial_window: int, horizon: int = 1, step: int = 1) -> dict:
    values = np.asarray(prices, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any()
            or type(initial_window) is not int or initial_window < 2
            or type(horizon) is not int or horizon < 1
            or type(step) is not int or step < 1 or initial_window + horizon > len(values)):
        raise ValueError("Invalid price history or evaluation windows")
    predictions = []
    for end in range(initial_window, len(values) - horizon + 1, step):
        context = values[:end]
        forecast = forecast_evidence(context, horizon, forecaster)
        predicted = float(context[-1] * (1 + forecast['median_returns'][-1]))
        predictions.append({'origin_index': end - 1, 'target_index': end + horizon - 1,
                            'actual': float(values[end + horizon - 1]),
                            'model': predicted, 'naive': float(context[-1])})
    model_errors = np.array([row['model'] - row['actual'] for row in predictions])
    naive_errors = np.array([row['naive'] - row['actual'] for row in predictions])
    return {'horizon': horizon, 'samples': len(predictions), 'predictions': predictions,
            'model_mae': float(np.mean(np.abs(model_errors))),
            'naive_mae': float(np.mean(np.abs(naive_errors))),
            'model_rmse': float(np.sqrt(np.mean(model_errors ** 2))),
            'naive_rmse': float(np.sqrt(np.mean(naive_errors ** 2)))}
