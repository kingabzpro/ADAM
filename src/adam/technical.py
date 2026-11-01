import numpy as np
import pandas as pd


def _wilder(values, period):
    average = values.iloc[:period].mean()
    for value in values.iloc[period:]:
        average = (average * (period - 1) + value) / period
    return float(average)


def technical_evidence(history):
    columns = ["High", "Low", "Close", "Volume"]
    if (len(history) < 35 or not set(columns).issubset(history.columns)
            or not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None
            or not history.index.is_monotonic_increasing or history.index.has_duplicates):
        raise ValueError("At least 35 ordered, dated OHLCV observations are required")
    try:
        data = history[columns].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("OHLCV values must be numeric") from error
    high, low, close, volume = (data[column] for column in columns)
    if (not np.isfinite(data.to_numpy()).all() or (low <= 0).any() or (volume < 0).any()
            or (high < close).any() or (close < low).any()):
        raise ValueError("Invalid OHLCV values")
    changes = close.diff().iloc[1:]
    gain = _wilder(changes.clip(lower=0), 14)
    loss = _wilder(-changes.clip(upper=0), 14)
    rsi = 50.0 if gain == loss == 0 else 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    recent = close.iloc[-20:]
    mean = recent.mean()
    deviation = recent.std(ddof=0)
    previous = close.shift(1)
    true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    average_volume = volume.iloc[-20:].mean()
    return {
        "as_of": history.index[-1].isoformat(), "rsi_14": float(rsi),
        "macd": float(macd.iloc[-1]), "macd_signal": float(macd_signal.iloc[-1]),
        "sma_20": float(mean), "ema_20": float(close.ewm(span=20, adjust=False).mean().iloc[-1]),
        "bollinger_upper": float(mean + 2 * deviation), "bollinger_lower": float(mean - 2 * deviation),
        "atr_14": _wilder(true_range, 14), "momentum_10": float(close.iloc[-1] / close.iloc[-11] - 1),
        "volume_ratio_20": float(volume.iloc[-1] / average_volume) if average_volume > 0 else None,
        "support_20": float(low.iloc[-20:].min()), "resistance_20": float(high.iloc[-20:].max()),
    }


async def assess_technical(ticker, history, model):
    return await model.assess(ticker, "technical", technical_evidence(history))
