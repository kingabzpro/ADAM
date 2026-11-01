import asyncio

import yfinance as yf


def fetch_history(ticker: str, period: str = "1mo"):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be blank")
    return yf.Ticker(ticker).history(period=period, interval="1d", raise_errors=True)

async def fetch_many(tickers: list[str], concurrency: int = 4):
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(ticker: str):
        async with semaphore:
            return await asyncio.to_thread(fetch_history, ticker)

    return await asyncio.gather(*(fetch(ticker) for ticker in tickers), return_exceptions=True)


def market_evidence(history):
    import numpy as np
    import pandas as pd

    if (len(history) < 3 or not {"Close", "Volume"}.issubset(history.columns)
            or not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None
            or not history.index.is_monotonic_increasing or history.index.has_duplicates):
        raise ValueError("At least three ordered, dated price and volume observations are required")
    try:
        close = pd.to_numeric(history["Close"], errors="raise").astype(float)
        volume = pd.to_numeric(history["Volume"], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Price and volume must be numeric") from error
    if (not np.isfinite(close).all() or not np.isfinite(volume).all()
            or (close <= 0).any() or (volume < 0).any()):
        raise ValueError("Price and volume must be finite and valid")
    returns = close.pct_change().iloc[1:]
    return {
        "as_of": history.index[-1].isoformat(),
        "last_close": float(close.iloc[-1]),
        "last_volume": float(volume.iloc[-1]),
        "last_daily_return": float(returns.iloc[-1]),
        "period_return": float(close.iloc[-1] / close.iloc[0] - 1),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "observations": len(close),
    }


async def assess_market(ticker, history, model):
    return await model.assess(ticker, "market", market_evidence(history))


def compare_market(history, benchmark, *, benchmark_symbol='SPY'):
    import numpy as np
    import pandas as pd

    series = []
    for frame in (history, benchmark):
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None
                or not frame.index.is_monotonic_increasing or frame.index.has_duplicates
                or 'Close' not in frame.columns or len(frame) < 20):
            raise ValueError('Ordered dated close prices are required for market comparison')
        try:
            close = pd.to_numeric(frame['Close'], errors='raise').astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError('Invalid comparison prices') from error
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError('Comparison prices must be positive and finite')
        dates = pd.Index(frame.index.date)
        if dates.has_duplicates:
            raise ValueError('Only one close per market date is supported')
        series.append(pd.Series(close.to_numpy(), index=dates))
    if series[0].index[-1] != series[1].index[-1]:
        raise ValueError('Latest asset and benchmark sessions do not match')
    aligned = pd.concat(series, axis=1, join='inner')
    if len(aligned) < 20:
        raise ValueError('At least 20 matching market sessions are required')
    asset, market = aligned.iloc[:, 0], aligned.iloc[:, 1]
    asset_return = float(asset.iloc[-1] / asset.iloc[0] - 1)
    benchmark_return = float(market.iloc[-1] / market.iloc[0] - 1)
    average = float(market.iloc[-20:].mean())
    last = float(market.iloc[-1])
    return {'benchmark': benchmark_symbol, 'as_of': aligned.index[-1].isoformat(),
            'aligned_observations': len(aligned), 'asset_return': asset_return,
            'benchmark_return': benchmark_return, 'relative_return': asset_return - benchmark_return,
            'benchmark_sma_20': average,
            'benchmark_vs_sma_20': 'above' if last > average else 'below' if last < average else 'at'}
