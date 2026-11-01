import asyncio
import importlib.util
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from adam.coordinator import Coordinator
from adam.d3 import D3Client
from adam.forecast import shared_forecaster
from adam.health import health_message, ollama_health
from adam.history import record_analysis
from adam.news import news_evidence
from adam.presentation import evidence_summary, safe_news_link, vote_summary
from adam.risk import RiskEvidence, RiskLimits
from adam.sentiment import shared_classifier
from adam.snapshots import read_news


HISTORY_PATH = Path.home() / '.adam' / 'history.sqlite'
SNAPSHOT_DIRECTORY = Path.home() / '.adam' / 'snapshots'

# Conservative built-in limits for portfolio-free research. Market data uses
# daily bars, so freshness tolerates a weekend; supply your own limits to
# tighten any threshold (docs/risk-input.md).
DEFAULT_LIMITS = RiskLimits(max_age_seconds=4 * 86400, max_volatility=0.5, max_downside=0.2,
                            max_position_fraction=0.1, max_concentration=0.3, min_daily_dollar_volume=1000000,
                            max_correlated_exposure=0.4, min_reward_risk=2, max_transaction_cost_fraction=0.01,
                            max_loss_fraction=0.05)


def parse_tickers(value: str) -> list[str]:
    return list(dict.fromkeys(ticker.strip().upper() for ticker in value.split(",") if ticker.strip()))


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


def agent_signals(result) -> dict:
    signals = {}
    for name, status in result.agents.items():
        decision = result.decisions.get(name)
        if decision is None:
            signals[name] = {'signal': None, 'confidence': None, 'abstained': True, 'status': status,
                             'probabilities': None}
        else:
            signals[name] = {'signal': decision.signal, 'confidence': max(decision.probabilities.values()),
                             'abstained': not decision.reliable or decision.stale or decision.status != "completed",
                             'status': status,
                             'probabilities': {name: decision.probabilities[name] for name in ('BUY', 'HOLD', 'SELL')}}
    return signals


async def run_analysis(tickers: list[str], inference_concurrency: int, limits=None, evidence=None, *, forecast=False, news=False, on_event=None, on_result=None, history_path=None, data_concurrency=4, ticker_concurrency=4, offline=False, snapshot_directory=None, cache_ttl_seconds=0) -> list[dict]:
    if type(ticker_concurrency) is not int or ticker_concurrency < 1:
        raise ValueError('ticker_concurrency must be a positive integer')
    async with httpx.AsyncClient(timeout=180, trust_env=False) as http:
        forecaster = (await asyncio.to_thread(shared_forecaster, local_files_only=True) if offline
                      else await asyncio.to_thread(shared_forecaster)) if forecast else None
        classifier = await asyncio.to_thread(shared_classifier) if news and not offline else None
        news_http = httpx.AsyncClient(timeout=30, trust_env=False) if news and not offline else None

        async def news_provider(ticker):
            if offline:
                evidence, saved = await asyncio.to_thread(read_news, snapshot_directory, ticker)
                evidence.update(source='local snapshot', snapshot_saved_at=saved.isoformat())
                return evidence
            return await news_evidence(ticker, news_http, classifier)

        coordinator = Coordinator(D3Client(http), limits, forecaster=forecaster,
                                  news_provider=news_provider if news else None,
                                  inference_concurrency=inference_concurrency, data_concurrency=data_concurrency,
                                  offline=offline, snapshot_directory=snapshot_directory,
                                  cache_ttl_seconds=cache_ttl_seconds)
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
            vote = vote_summary(list(result.decisions.values()))
            counts = vote['counts']
            vote_display = f"{vote['signal'] or 'No consensus'}: BUY {counts['BUY']}, HOLD {counts['HOLD']}, SELL {counts['SELL']}"
            row = {"ticker": ticker, "agents": "; ".join(agents), "vote": vote, "vote_display": vote_display,
                   "signals": agent_signals(result),
                   "evidence": result.evidence, "risk_inputs": risk_inputs, "action": result.risk.action,
                   "risk": f"{result.risk.status}: {', '.join(result.risk.reasons)}",
                   "errors": "; ".join(f"{name}: {message}" for name, message in errors.items())}
            if on_result is not None:
                on_result(row)
            return row

        try:
            return list(await asyncio.gather(*(analyze(ticker) for ticker in tickers)))
        finally:
            if news_http is not None:
                await news_http.aclose()


def result_payload(row: dict) -> dict:
    news = row['evidence'].get('news') or {}
    links = []
    for item in news.get(news.get('basis', ''), []):
        links.append({'title': item.get('title', 'News'), 'url': safe_news_link(item.get('url')),
                      'published': item.get('published'), 'sentiment': (item.get('sentiment') or {}).get('label', 'unavailable'),
                      'summary': item.get('summary'), 'highlights': item.get('financial_highlights', [])})
    forecast = row['evidence'].get('forecast') or {}
    quantiles = {name: forecast.get(name) for name in
                 ('horizons', 'lower_returns', 'median_returns', 'upper_returns')}
    return {'ticker': row['ticker'], 'agents': row['agents'], 'signals': row['signals'], 'vote': row['vote'],
            'vote_display': row['vote_display'], 'errors': row['errors'],
            'summary': evidence_summary(row['evidence']), 'forecast': quantiles, 'news_links': links}


def create_app(config, *, offline=False, snapshot_directory=SNAPSHOT_DIRECTORY, history_path=HISTORY_PATH,
               forecast=None, news=None) -> FastAPI:
    page = Path(__file__).parent / 'static' / 'index.html'
    if forecast is None:
        forecast = importlib.util.find_spec('timestral') is not None
    if news is None:
        news = importlib.util.find_spec('transformers') is not None
    app = FastAPI(title='ADAM', docs_url=None, redoc_url=None)

    @app.get('/')
    async def serve_page():
        # no-cache keeps the UI in sync with code changes; browsers otherwise heuristically cache the page.
        return FileResponse(page, headers={'Cache-Control': 'no-cache'})

    @app.get('/favicon.ico')
    async def favicon():
        icon = Path(__file__).parent / 'static' / 'favicon.svg'
        return FileResponse(icon, media_type='image/svg+xml', headers={'Cache-Control': 'max-age=86400'})

    @app.get('/api/health')
    async def health():
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            status = await ollama_health(client)
        if status['status'] != 'ready':
            return {'ready': False, 'message': health_message(status)}
        missing = [name for name, module in (('forecast', 'timestral'), ('sentiment', 'transformers'))
                   if importlib.util.find_spec(module) is None]
        if missing:
            return {'ready': False, 'message': 'Ollama and d3:2b are ready, but the ' + ' and '.join(missing) +
                    ' extras are missing. Run uv sync --extra forecast --extra sentiment.'}
        return {'ready': True, 'message': 'All models loaded'}

    @app.get('/api/analysis')
    async def analysis(tickers: str):
        symbols = parse_tickers(tickers)
        if not symbols or len(symbols) > 20:
            raise HTTPException(status_code=400, detail='Enter between 1 and 20 valid ticker symbols')
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(event):
            queue.put_nowait(('event', event))

        def on_result(row):
            queue.put_nowait(('result', result_payload(row)))

        async def run():
            try:
                await run_analysis(symbols, config.inference_concurrency, DEFAULT_LIMITS, forecast=forecast,
                                   news=news, on_event=on_event, on_result=on_result, history_path=history_path,
                                   data_concurrency=config.data_concurrency,
                                   ticker_concurrency=config.ticker_concurrency, offline=offline,
                                   snapshot_directory=snapshot_directory,
                                   cache_ttl_seconds=config.cache_ttl_seconds)
            except Exception as error:
                queue.put_nowait(('error', {'message': str(error)}))
            finally:
                queue.put_nowait(None)

        async def stream():
            task = asyncio.create_task(run())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        yield 'event: done\ndata: {}\n\n'
                        return
                    kind, payload = item
                    yield f'event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'
            finally:
                task.cancel()
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache'})
    return app


def run_dashboard(config) -> None:
    import uvicorn

    offline = os.environ.get('ADAM_OFFLINE', '').strip().lower() in ('1', 'true', 'yes')
    app = create_app(config, offline=offline)
    print('ADAM dashboard: http://127.0.0.1:8080')
    uvicorn.run(app, host='127.0.0.1', port=8080, log_level='warning')
