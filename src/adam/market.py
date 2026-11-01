import asyncio

import yfinance as yf


def fetch_history(ticker: str):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be blank")
    return yf.Ticker(ticker).history(period="1mo", interval="1d", raise_errors=True)

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
