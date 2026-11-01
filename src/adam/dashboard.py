import asyncio
import json
import importlib.util
from pathlib import Path

import httpx
from nicegui import ui

from adam.coordinator import Coordinator
from adam.d3 import D3Client
from adam.evaluation import evaluate_saved_buys
from adam.forecast import shared_forecaster
from adam.health import health_message, ollama_health
from adam.history import record_analysis, read_history
from adam.news import news_evidence
from adam.sentiment import shared_classifier
from adam.risk import RiskEvidence, RiskLimits
from adam.snapshots import read_news, read_prices

from adam.market import fetch_history, fetch_many


def parse_tickers(value: str) -> list[str]:
    return list(dict.fromkeys(ticker.strip().upper() for ticker in value.split(",") if ticker.strip()))


async def saved_buy_report(ticker, history_path):
    records = await asyncio.to_thread(read_history, history_path, ticker=ticker, limit=1000)
    prices = await asyncio.to_thread(fetch_history, ticker, '1y')
    return evaluate_saved_buys(records, prices, ticker=ticker)


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


def format_agent(name, status, decision):
    if decision is None:
        return f"{name}: {status}"
    scores = ', '.join(f"{signal} {decision.probabilities[signal]:.1%}" for signal in ('BUY', 'HOLD', 'SELL'))
    label = " [abstained]" if not decision.reliable or decision.stale or decision.status != "completed" else ""
    return f"{name}: {decision.signal} ({scores}){label}"


async def run_analysis(tickers: list[str], inference_concurrency: int, limits=None, evidence=None, *, forecast=False, news=False, on_event=None, history_path=None, data_concurrency=4, ticker_concurrency=4, offline=False, snapshot_directory=None) -> list[dict]:
    if type(ticker_concurrency) is not int or ticker_concurrency < 1:
        raise ValueError('ticker_concurrency must be a positive integer')
    async with httpx.AsyncClient(timeout=180, trust_env=False) as http:
        forecaster = (await asyncio.to_thread(shared_forecaster, local_files_only=True) if offline
                      else await asyncio.to_thread(shared_forecaster)) if forecast else None
        classifier = await asyncio.to_thread(shared_classifier) if news and not offline else None

        async def news_provider(ticker):
            if offline:
                evidence, saved = await asyncio.to_thread(read_news, snapshot_directory, ticker)
                evidence.update(source='local snapshot', snapshot_saved_at=saved.isoformat())
                return evidence
            async with httpx.AsyncClient(timeout=30, trust_env=False) as news_http:
                return await news_evidence(ticker, news_http, classifier)

        coordinator = Coordinator(D3Client(http), limits, forecaster=forecaster,
                                  news_provider=news_provider if news else None,
                                  inference_concurrency=inference_concurrency, data_concurrency=data_concurrency,
                                  offline=offline, snapshot_directory=snapshot_directory)
        ticker_slots = asyncio.Semaphore(ticker_concurrency)

        async def analyze(ticker):
            async with ticker_slots:
                ticker_evidence = (evidence or {}).get(ticker, RiskEvidence())
                result = await coordinator.analyze(ticker, ticker_evidence, on_event=on_event)
            errors = dict(result.errors)
            if history_path is not None:
                try:
                    await asyncio.to_thread(record_analysis, history_path, result, risk_evidence=ticker_evidence, limits=limits)
                except Exception as error:
                    errors['history'] = str(error)
            agents = []
            for name, status in result.agents.items():
                decision = result.decisions.get(name)
                agents.append(format_agent(name, status, decision))
            evaluated = getattr(result, 'evaluated_risk_evidence', ticker_evidence)
            risk_inputs = {'supplied': ticker_evidence.model_dump(mode='json'), 'evaluated': evaluated.model_dump(mode='json')}
            return {"ticker": ticker, "agents": "; ".join(agents), "evidence": result.evidence, "risk_inputs": risk_inputs, "action": result.risk.action,
                    "risk": f"{result.risk.status}: {', '.join(result.risk.reasons)}",
                    "errors": "; ".join(f"{name}: {message}" for name, message in errors.items())}

        return await asyncio.gather(*(analyze(ticker) for ticker in tickers))


def dashboard(config) -> None:
    ui.label("ADAM").classes("text-h4")
    ui.label("Local market research and decision support")
    ui.label("Research signals are preliminary. No Trade is returned until risk limits and evidence are configured.")
    health_status = ui.label('Checking local Ollama...')

    async def check_health():
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            health_status.text = health_message(await ollama_health(client))

    ui.timer(0.1, check_health, once=True)
    ui.button('Check model readiness', on_click=check_health)
    snapshot_directory = Path.home() / '.adam' / 'snapshots'
    offline_toggle = ui.checkbox('Offline: use saved snapshots, always No Trade', value=False)
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
            if offline_toggle.value:
                results = await asyncio.gather(*(asyncio.to_thread(read_prices, snapshot_directory, symbol)
                                                 for symbol in symbols), return_exceptions=True)
                results = [item[0] if not isinstance(item, BaseException) else item for item in results]
            else:
                results = await fetch_many(symbols)
            table.rows = market_rows(symbols, results)
            status.text = 'Saved snapshots; dates may be stale.' if offline_toggle.value else "Daily closing prices, adjusted by Yahoo Finance."
        finally:
            button.enable()

    button = ui.button("Load prices", on_click=load)
    ui.separator()
    ui.label("Agent analysis").classes("text-h6")
    history_path = Path.home() / '.adam' / 'history.sqlite'
    ui.label(f'Analyses and supplied risk inputs are saved locally to {history_path}.')
    analysis_status = ui.label("Requires Ollama and the local d3:2b model. Run uv run adam-setup.")
    forecast_toggle = ui.checkbox('Include Timestral forecast (requires forecast extra)', value=importlib.util.find_spec('timestral') is not None)
    news_toggle = ui.checkbox('Include news sentiment (requires sentiment extra)', value=False)
    risk_input = ui.textarea("Optional risk limits and per-ticker evidence (JSON)").classes("w-full")
    ui.label('Analysis timeline')
    timeline = ui.log(max_lines=200).classes('w-full h-40')
    analysis_table = ui.table(columns=[
        {"name": "ticker", "label": "Ticker", "field": "ticker"},
        {"name": "agents", "label": "Agent signals and model scores", "field": "agents"},
        {"name": "action", "label": "Risk-gated result", "field": "action"},
        {"name": "risk", "label": "Risk status", "field": "risk"},
        {"name": "errors", "label": "Errors", "field": "errors"},
    ], rows=[], row_key="ticker")

    evidence_details = ui.column().classes('w-full')

    async def analyze():
        symbols = parse_tickers(tickers.value or "")
        if not symbols or len(symbols) > 20:
            analysis_status.text = "Enter between 1 and 20 tickers."
            return
        analyze_button.disable()
        analysis_status.text = "Analyzing..."
        analysis_table.rows = []
        timeline.clear()
        evidence_details.clear()

        def show_event(event):
            timeline.push(f"{event['time']} {event['ticker']} {event['stage']}: {event['status']}")

        try:
            limits, evidence = parse_risk_input(risk_input.value or "")
            analysis_table.rows = await run_analysis(symbols, config.inference_concurrency, limits, evidence, forecast=forecast_toggle.value, news=news_toggle.value, on_event=show_event, history_path=history_path, data_concurrency=config.data_concurrency, ticker_concurrency=config.ticker_concurrency, offline=offline_toggle.value, snapshot_directory=snapshot_directory)
            with evidence_details:
                for row in analysis_table.rows:
                    with ui.expansion(f"{row['ticker']} evidence").classes('w-full'):
                        ui.code(json.dumps(row['evidence'], indent=2, ensure_ascii=False), language='json').classes('w-full')
                        ui.label('Supplied and evaluated risk inputs')
                        ui.code(json.dumps(row['risk_inputs'], indent=2, ensure_ascii=False), language='json').classes('w-full')
            analysis_status.text = "Model scores are not probabilities of profit."
        except Exception as error:
            analysis_status.text = f"Analysis failed: {error}"
        finally:
            analyze_button.enable()

    analyze_button = ui.button("Analyze", on_click=analyze)
    ui.separator()
    ui.label('Saved analysis history').classes('text-h6')
    history_status = ui.label()
    history_table = ui.table(columns=[
        {'name': 'time', 'label': 'Saved at (UTC)', 'field': 'time'},
        {'name': 'ticker', 'label': 'Ticker', 'field': 'ticker'},
        {'name': 'action', 'label': 'Risk-gated result', 'field': 'action'},
        {'name': 'risk', 'label': 'Risk status', 'field': 'risk'},
    ], rows=[], row_key='id')

    saved_records = {}
    history_details = ui.column().classes('w-full')

    def select_history(event):
        history_details.clear()
        record = saved_records.get(event.value)
        if record is not None:
            with history_details:
                ui.code(json.dumps(record, indent=2, ensure_ascii=False), language='json').classes('w-full')

    history_selector = ui.select({}, label='View saved analysis', on_change=select_history).classes('w-full')

    async def show_history():
        try:
            records = await asyncio.to_thread(read_history, history_path)
            history_table.rows = [{'id': record['id'], 'time': record['created_at'], 'ticker': record['ticker'],
                                   'action': record['risk']['action'], 'risk': record['risk']['status']}
                                  for record in records]
            saved_records.clear()
            saved_records.update({str(record['id']): record for record in records})
            history_selector.set_options({str(record['id']): f"{record['created_at']} — {record['ticker']}" for record in records}, value=None)
            history_details.clear()
            history_status.text = f'{len(records)} recent analyses'
        except Exception as error:
            history_status.text = f'Unable to read history: {error}'

    ui.button('Load history', on_click=show_history)
    ui.label('Hypothetical Buy outcomes: next session open to fifth completed session close, before costs. Overlapping outcomes are not portfolio returns.')
    outcomes_status = ui.label()
    outcomes = ui.code('', language='json').classes('w-full')

    async def show_outcomes():
        if offline_toggle.value:
            outcomes_status.text = 'Outcome evaluation requires online market data.'
            return
        symbols = parse_tickers(tickers.value or '')
        if len(symbols) != 1:
            outcomes_status.text = 'Enter exactly one ticker to evaluate saved Buy decisions.'
            return
        outcomes_button.disable()
        outcomes_status.text = 'Loading saved decisions and prices...'
        try:
            report = await saved_buy_report(symbols[0], history_path)
            outcomes.content = json.dumps(report, indent=2, ensure_ascii=False)
            outcomes_status.text = f"{report['completed_count']} completed outcomes from up to 1,000 saved analyses."
        except Exception as error:
            outcomes_status.text = f'Unable to evaluate saved decisions: {error}'
        finally:
            outcomes_button.enable()

    outcomes_button = ui.button('Evaluate saved Buy decisions', on_click=show_outcomes)


def run_dashboard(config) -> None:
    ui.run(lambda: dashboard(config), title="ADAM", host="127.0.0.1", reload=False)
