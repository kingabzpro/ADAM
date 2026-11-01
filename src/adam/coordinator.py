import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass

from adam.decisions import Decision, consensus
from adam.forecast import forecast_evidence
from adam.fundamentals import fetch_fundamentals, fundamentals_evidence
from adam.market import compare_market, fetch_history, market_evidence
from adam.news import compact_news_evidence
from adam.risk import RiskEvidence, RiskLimits, RiskResult, derive_risk_evidence, evaluate_risk
from adam.technical import technical_evidence


@dataclass
class AnalysisResult:
    ticker: str
    agents: dict[str, str]
    decisions: dict[str, Decision]
    risk: RiskResult
    errors: dict[str, str]
    timeline: list[dict[str, str]]
    evidence: dict[str, dict]
    evaluated_risk_evidence: RiskEvidence


class Coordinator:
    def __init__(self, model, limits: RiskLimits | None, *, forecaster=None, news_provider=None, inference_concurrency: int = 1, data_concurrency: int = 4, offline: bool = False, snapshot_directory=None):
        if type(inference_concurrency) is not int or inference_concurrency < 1:
            raise ValueError("inference_concurrency must be a positive integer")
        if type(data_concurrency) is not int or data_concurrency < 1:
            raise ValueError("data_concurrency must be a positive integer")
        if type(offline) is not bool or (offline and snapshot_directory is None):
            raise ValueError('Offline mode requires a snapshot directory')
        self.model = model
        self.limits = limits
        self.forecaster = forecaster
        self.news_provider = news_provider
        self.offline = offline
        self.snapshot_directory = snapshot_directory
        self.snapshot_times = {}
        self.snapshot_errors = {}
        self.inference_slots = asyncio.Semaphore(inference_concurrency)
        self.forecast_slot = asyncio.Semaphore(1)
        self.data_slots = asyncio.Semaphore(data_concurrency)
        self._data_tasks = {}
        self._benchmark_task = None
        self._benchmark_started = 0.0

    def _load_prices(self, ticker, period):
        from adam.snapshots import read_prices, save_prices
        if self.offline:
            data, saved = read_prices(self.snapshot_directory, ticker)
            self.snapshot_times[('prices', ticker)] = saved.isoformat()
            return data
        data = fetch_history(ticker, period)
        if self.snapshot_directory is not None and period == '3mo':
            try:
                save_prices(self.snapshot_directory, ticker, data)
                self.snapshot_errors.pop(('prices', ticker), None)
            except (OSError, ValueError, TypeError) as error:
                self.snapshot_errors[('prices', ticker)] = str(error)
                logging.exception('Unable to save price snapshot for %s', ticker)
        return data

    def _load_fundamentals(self, ticker):
        from adam.snapshots import read_fundamentals, save_fundamentals
        if self.offline:
            data, saved = read_fundamentals(self.snapshot_directory, ticker)
            self.snapshot_times[('fundamentals', ticker)] = saved.isoformat()
            return data
        data = fetch_fundamentals(ticker)
        if self.snapshot_directory is not None:
            try:
                save_fundamentals(self.snapshot_directory, ticker, data)
                self.snapshot_errors.pop(('fundamentals', ticker), None)
            except (OSError, ValueError, TypeError) as error:
                self.snapshot_errors[('fundamentals', ticker)] = str(error)
                logging.exception('Unable to save fundamentals snapshot for %s', ticker)
        return data

    async def _fetch_data(self, key, function, *args):
        task = self._data_tasks.get(key)
        if task is None or task.done():
            async def fetch():
                async with self.data_slots:
                    return await asyncio.to_thread(function, *args)
            task = asyncio.create_task(fetch())
            self._data_tasks[key] = task

            def finished(done):
                if self._data_tasks.get(key) is done:
                    del self._data_tasks[key]
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def benchmark_history(self):
        task = self._benchmark_task
        if (task is None or (task.done() and (task.cancelled() or task.exception() is not None
                                               or time.monotonic() - self._benchmark_started > 60))):
            self._benchmark_started = time.monotonic()
            task = asyncio.create_task(self._fetch_data(('prices', 'SPY'), self._load_prices, 'SPY', '3mo'))
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            self._benchmark_task = task
        return await asyncio.shield(task)

    async def analyze(self, ticker: str, risk_evidence: RiskEvidence, *, horizon: int = 5, on_event=None) -> AnalysisResult:
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker is required")
        agents = {name: "unavailable" for name in ("market", "technical", "forecast", "news", "fundamentals")}
        decisions = {}
        errors = {}
        timeline = []
        evidence_by_agent = {}

        def emit(stage, status):
            event = {'time': datetime.now(timezone.utc).isoformat(), 'ticker': ticker,
                     'stage': stage, 'status': status}
            timeline.append(event)
            if on_event is not None:
                try:
                    on_event(event.copy())
                except Exception:
                    logging.exception('Analysis event callback failed')

        async def run(name, compute):
            emit(name, "started")
            try:
                evidence = await compute()
                evidence_by_agent[name] = evidence
                model_evidence = compact_news_evidence(evidence) if name == 'news' else evidence
                async with self.inference_slots:
                    result = await self.model.assess(ticker, name, model_evidence)
                if result.ticker != ticker or result.agent != name or result.status != "completed":
                    raise ValueError("Mismatched agent response")
                if name == 'market' and evidence.get('comparison_status') != 'available':
                    result = result.model_copy(update={'reliable': False})
                if name == 'news' and self.offline:
                    result = result.model_copy(update={'stale': True})
                decisions[name] = result
                agents[name] = "completed"
                emit(name, "completed")
            except Exception as error:
                agents[name] = "failed"
                errors[name] = str(error)
                emit(name, "failed")

        async def fundamentals():
            info = await self._fetch_data(('fundamentals', ticker), self._load_fundamentals, ticker)
            evidence = fundamentals_evidence(info)
            if self.offline:
                evidence.update(source='local snapshot', snapshot_saved_at=self.snapshot_times[('fundamentals', ticker)])
            if not evidence["metrics"] or evidence["freshness"] != "current":
                raise ValueError("Fundamentals missing or fiscal quarter date stale or unknown")
            return evidence

        async def market_stream():
            emit("market_data", "started")
            try:
                history = await self._fetch_data(('prices', ticker), self._load_prices, ticker, '3mo')
            except Exception as error:
                errors["market_data"] = str(error)
                emit("market_data", "failed")
                agents["market"] = agents["technical"] = "failed"
                if self.forecaster is not None:
                    agents["forecast"] = "failed"
                return

            emit("market_data", "completed")

            async def market():
                evidence = await asyncio.to_thread(market_evidence, history)
                if self.offline:
                    evidence.update(source='local snapshot', snapshot_saved_at=self.snapshot_times[('prices', ticker)])
                emit('benchmark', 'started')
                try:
                    benchmark = history if ticker == 'SPY' else await self.benchmark_history()
                    evidence['comparison'] = await asyncio.to_thread(compare_market, history, benchmark)
                    evidence['comparison_status'] = 'available'
                    if self.offline:
                        evidence['benchmark_snapshot_saved_at'] = self.snapshot_times[('prices', 'SPY')]
                    emit('benchmark', 'completed')
                except Exception as error:
                    evidence['comparison'] = None
                    evidence['comparison_status'] = 'unavailable'
                    errors['benchmark'] = str(error)
                    emit('benchmark', 'failed')
                return evidence

            async def technical():
                return await asyncio.to_thread(technical_evidence, history)

            async def forecast():
                async with self.forecast_slot:
                    return await asyncio.to_thread(forecast_evidence, history["Close"].to_numpy(), horizon, self.forecaster)

            tasks = [run("market", market), run("technical", technical)]
            if self.forecaster is not None:
                tasks.append(run("forecast", forecast))
            await asyncio.gather(*tasks)

        async def news():
            evidence = await self.news_provider(ticker)
            if not (evidence.get('headlines') or evidence.get('articles')) or evidence.get('mean_scores') is None:
                raise ValueError('No recent news available')
            if not self.offline and self.snapshot_directory is not None:
                from adam.snapshots import save_news
                try:
                    await asyncio.to_thread(save_news, self.snapshot_directory, ticker, evidence)
                except (OSError, ValueError, TypeError) as error:
                    errors['snapshot_news'] = str(error)
            return evidence

        tasks = [market_stream(), run("fundamentals", fundamentals)]
        if self.news_provider is not None:
            tasks.append(run('news', news))
        await asyncio.gather(*tasks)
        for kind, symbol in (('prices', ticker), ('fundamentals', ticker), ('prices', 'SPY')):
            if (kind, symbol) in self.snapshot_errors:
                label = f'snapshot_{kind}' if symbol == ticker else 'snapshot_benchmark'
                errors[label] = self.snapshot_errors[(kind, symbol)]
        evaluated_risk = derive_risk_evidence(risk_evidence, evidence_by_agent.get('market'), evidence_by_agent.get('forecast'))
        result = (evaluate_risk(consensus(list(decisions.values())), evaluated_risk, self.limits)
                  if self.limits is not None else RiskResult(status="insufficient_evidence", action="NO_TRADE", reasons=("risk_limits",)))
        missing_critical = tuple(f"{name}_decision" for name in ("market", "forecast")
                                 if name not in decisions or not decisions[name].reliable or decisions[name].stale)
        if missing_critical:
            result = RiskResult(status="fail" if result.status == "fail" else "insufficient_evidence", action="NO_TRADE",
                                reasons=result.reasons + missing_critical)
        if self.offline:
            result = RiskResult(status='fail' if result.status == 'fail' else 'insufficient_evidence',
                                action='NO_TRADE', reasons=result.reasons + ('offline_snapshot',))
        emit("risk", result.action)
        return AnalysisResult(ticker, agents, decisions, result, errors, timeline, evidence_by_agent, evaluated_risk)
