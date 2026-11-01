import yfinance as yf


def fetch_history(ticker: str):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be blank")
    return yf.Ticker(ticker).history(period="1mo", interval="1d", raise_errors=True)
