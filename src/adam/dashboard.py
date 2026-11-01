from nicegui import ui

from adam.market import fetch_many


def parse_tickers(value: str) -> list[str]:
    return list(dict.fromkeys(ticker.strip().upper() for ticker in value.split(",") if ticker.strip()))


def market_rows(tickers, results):
    rows = []
    for ticker, result in zip(tickers, results, strict=True):
        row = {"ticker": ticker, "date": "", "close": "", "status": "Data unavailable"}
        if not isinstance(result, BaseException):
            if result.empty or "Close" not in result:
                row["status"] = "No price history"
            else:
                closes = result["Close"].dropna()
                if closes.empty:
                    row["status"] = "No price history"
                else:
                    row.update(date=str(closes.index[-1].date()), close=f"{closes.iloc[-1]:.2f}", status="Daily close")
        rows.append(row)
    return rows


def dashboard() -> None:
    ui.label("ADAM").classes("text-h4")
    ui.label("Local market research and decision support")
    ui.label("Historical daily closes; no recommendation is available yet.")
    tickers = ui.input("Tickers, separated by commas", value="AAPL, MSFT")
    status = ui.label()
    table = ui.table(columns=[
        {"name": "ticker", "label": "Ticker", "field": "ticker"},
        {"name": "date", "label": "Date", "field": "date"},
        {"name": "close", "label": "Close", "field": "close"},
        {"name": "status", "label": "Status", "field": "status"},
    ], rows=[], row_key="ticker")

    async def load():
        symbols = parse_tickers(tickers.value or "")
        if not symbols:
            status.text = "Enter at least one ticker."
            return
        button.disable()
        status.text = "Loading market data..."
        table.rows = []
        try:
            results = await fetch_many(symbols)
            table.rows = market_rows(symbols, results)
            status.text = "Daily closing prices, adjusted by Yahoo Finance."
        finally:
            button.enable()

    button = ui.button("Load prices", on_click=load)


def run_dashboard() -> None:
    ui.run(dashboard, title="ADAM", host="127.0.0.1", reload=False)
