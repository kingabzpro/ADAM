import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass

from adam.decisions import Decision, consensus
from adam.forecast import forecast_evidence
from adam.fundamentals import fetch_fundamentals, fundamentals_evidence
from adam.market import compare_market, fetch_history, market_evidence
from adam.risk import RiskEvidence, RiskLimits, RiskResult, evaluate_risk
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
    def __init__(self, model, limits: RiskLimits | None, *, forecaster=None, news_provider=None, inference_concurrency: int = 1):
        if type(inference_concurrency) is not int or inference_concurrency < 1:
            raise ValueError("inference_concurrency must be a positive integer")
        self.model = model
        self.limits = limits
        self.forecaster = forecaster
        self.news_provider = news_provider
        self.inference_slots = asyncio.Semaphore(inference_concurrency)
        self.forecast_slot = asyncio.Semaphore(1)
        self._benchmark_task = None
        self._benchmark_started = 0.0

    async def benchmark_history(self):
        task = self._benchmark_task
        if (task is None or (task.done() and (task.cancelled() or task.exception() is not None
                                               or time.monotonic() - self._benchmark_started > 60))):
            self._benchmark_started = time.monotonic()
            task = asyncio.create_task(asyncio.to_thread(fetch_history, 'SPY', '3mo'))
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
                async with self.inference_slots:
                    result = await self.model.assess(ticker, name, evidence)
                if result.ticker != ticker or result.agent != name or result.status != "completed":
                    raise ValueError("Mismatched agent response")
                if name == 'market' and evidence.get('comparison_status') != 'available':
                    result = result.model_copy(update={'reliable': False})
                decisions[name] = result
                agents[name] = "completed"
                emit(name, "completed")
            except Exception as error:
                agents[name] = "failed"
                errors[name] = str(error)
                emit(name, "failed")

        async def fundamentals():
            info = await asyncio.to_thread(fetch_fundamentals, ticker)
            evidence = fundamentals_evidence(info)
            if not evidence["metrics"] or evidence["freshness"] != "current":
                raise ValueError("Fundamentals missing or fiscal quarter date stale or unknown")
            return evidence

        async def market_stream():
            emit("market_data", "started")
            try:
                history = await asyncio.to_thread(fetch_history, ticker, "3mo")
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
                emit('benchmark', 'started')
                try:
                    benchmark = history if ticker == 'SPY' else await self.benchmark_history()
                    evidence['comparison'] = await asyncio.to_thread(compare_market, history, benchmark)
                    evidence['comparison_status'] = 'available'
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
            if not evidence.get('headlines') or evidence.get('mean_scores') is None:
                raise ValueError('No recent headlines available')
            return evidence

        tasks = [market_stream(), run("fundamentals", fundamentals)]
        if self.news_provider is not None:
            tasks.append(run('news', news))
        await asyncio.gather(*tasks)
        evaluated_risk = risk_evidence
        market = evidence_by_agent.get('market')
        if market and risk_evidence.market_time is not None:
            observed_time = datetime.fromisoformat(market['as_of'])
            evaluated_risk = risk_evidence.model_copy(update={
                'market_time': min(risk_evidence.market_time, observed_time),
            })
        result = (evaluate_risk(consensus(list(decisions.values())), evaluated_risk, self.limits)
                  if self.limits is not None else RiskResult(status="insufficient_evidence", action="NO_TRADE", reasons=("risk_limits",)))
        missing_critical = tuple(f"{name}_decision" for name in ("market", "forecast")
                                 if name not in decisions or not decisions[name].reliable or decisions[name].stale)
        if missing_critical and result.status != "fail":
            result = RiskResult(status="insufficient_evidence", action="NO_TRADE",
                                reasons=result.reasons + missing_critical)
        emit("risk", result.action)
        return AnalysisResult(ticker, agents, decisions, result, errors, timeline, evidence_by_agent, evaluated_risk)
