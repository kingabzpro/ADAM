import asyncio
import json

import httpx
from nicegui import ui

from adam.coordinator import Coordinator
from adam.d3 import D3Client
from adam.risk import RiskEvidence, RiskLimits

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


def parse_risk_input(text: str):
    if not text.strip():
        return None, {}
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) != {"limits", "tickers"} or not isinstance(data["tickers"], dict):
            raise ValueError("Expected limits and tickers objects")
        limits = RiskLimits.model_validate(data["limits"])
        evidence = {}
        for ticker, values in data["tickers"].items():
            normalized = ticker.strip().upper()
            if not normalized or normalized in evidence:
                raise ValueError("Ticker keys must be unique and nonempty")
            evidence[normalized] = RiskEvidence.model_validate(values)
        return limits, evidence
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid risk configuration") from error


async def run_analysis(tickers: list[str], inference_concurrency: int, limits=None, evidence=None) -> list[dict]:
    async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
        coordinator = Coordinator(D3Client(http), limits, inference_concurrency=inference_concurrency)
        ticker_slots = asyncio.Semaphore(4)

        async def analyze(ticker):
            async with ticker_slots:
                result = await coordinator.analyze(ticker, (evidence or {}).get(ticker, RiskEvidence()))
            agents = []
            for name, status in result.agents.items():
                decision = result.decisions.get(name)
                agents.append(f"{name}: {decision.signal}" if decision else f"{name}: {status}")
            return {"ticker": ticker, "agents": "; ".join(agents), "action": result.risk.action,
                    "risk": f"{result.risk.status}: {', '.join(result.risk.reasons)}",
                    "errors": "; ".join(f"{name}: {message}" for name, message in result.errors.items())}

        return await asyncio.gather(*(analyze(ticker) for ticker in tickers))


def dashboard(config) -> None:
    ui.label("ADAM").classes("text-h4")
    ui.label("Local market research and decision support")
    ui.label("Research signals are preliminary. No Trade is returned until risk limits and evidence are configured.")
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
    ui.separator()
    ui.label("Agent analysis").classes("text-h6")
    analysis_status = ui.label("Requires a local D3 server at 127.0.0.1:8000.")
    risk_input = ui.textarea("Optional risk limits and per-ticker evidence (JSON)").classes("w-full")
    analysis_table = ui.table(columns=[
        {"name": "ticker", "label": "Ticker", "field": "ticker"},
        {"name": "agents", "label": "Agent signals", "field": "agents"},
        {"name": "action", "label": "Risk-gated result", "field": "action"},
        {"name": "risk", "label": "Risk status", "field": "risk"},
        {"name": "errors", "label": "Errors", "field": "errors"},
    ], rows=[], row_key="ticker")

    async def analyze():
        symbols = parse_tickers(tickers.value or "")
        if not symbols or len(symbols) > 20:
            analysis_status.text = "Enter between 1 and 20 tickers."
            return
        analyze_button.disable()
        analysis_status.text = "Analyzing..."
        analysis_table.rows = []
        try:
            limits, evidence = parse_risk_input(risk_input.value or "")
            analysis_table.rows = await run_analysis(symbols, config.inference_concurrency, limits, evidence)
            analysis_status.text = "Model scores are not probabilities of profit."
        except Exception as error:
            analysis_status.text = f"Analysis failed: {error}"
        finally:
            analyze_button.enable()

    analyze_button = ui.button("Analyze", on_click=analyze)


def run_dashboard(config) -> None:
    ui.run(lambda: dashboard(config), title="ADAM", host="127.0.0.1", reload=False)
