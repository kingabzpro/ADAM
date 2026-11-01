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
