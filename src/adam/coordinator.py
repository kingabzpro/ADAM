import asyncio
from dataclasses import dataclass

from adam.decisions import Decision, consensus
from adam.forecast import forecast_evidence
from adam.market import fetch_history, market_evidence
from adam.risk import RiskEvidence, RiskLimits, RiskResult, evaluate_risk
from adam.technical import technical_evidence


@dataclass
class AnalysisResult:
    ticker: str
    agents: dict[str, str]
    decisions: dict[str, Decision]
    risk: RiskResult
    errors: dict[str, str]


class Coordinator:
    def __init__(self, model, limits: RiskLimits, *, forecaster=None, inference_concurrency: int = 1):
        if type(inference_concurrency) is not int or inference_concurrency < 1:
            raise ValueError("inference_concurrency must be a positive integer")
        self.model = model
        self.limits = limits
        self.forecaster = forecaster
        self.inference_slots = asyncio.Semaphore(inference_concurrency)
        self.forecast_slot = asyncio.Semaphore(1)

    async def analyze(self, ticker: str, risk_evidence: RiskEvidence, *, horizon: int = 5) -> AnalysisResult:
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker is required")
        agents = {name: "unavailable" for name in ("market", "technical", "forecast", "news", "fundamentals")}
        decisions = {}
        errors = {}
        try:
            history = await asyncio.to_thread(fetch_history, ticker, "3mo")
        except Exception as error:
            errors["market_data"] = str(error)
            agents["market"] = agents["technical"] = "failed"
            if self.forecaster is not None:
                agents["forecast"] = "failed"
            return AnalysisResult(ticker, agents, decisions, evaluate_risk(consensus([]), risk_evidence, self.limits), errors)

        async def run(name, compute):
            try:
                evidence = await compute()
                async with self.inference_slots:
                    result = await self.model.assess(ticker, name, evidence)
                if result.ticker != ticker or result.agent != name or result.status != "completed":
                    raise ValueError("Mismatched agent response")
                decisions[name] = result
                agents[name] = "completed"
            except Exception as error:
                agents[name] = "failed"
                errors[name] = str(error)

        async def market():
            return await asyncio.to_thread(market_evidence, history)

        async def technical():
            return await asyncio.to_thread(technical_evidence, history)

        async def forecast():
            async with self.forecast_slot:
                return await asyncio.to_thread(forecast_evidence, history["Close"].to_numpy(), horizon, self.forecaster)

        tasks = [run("market", market), run("technical", technical)]
        if self.forecaster is not None:
            tasks.append(run("forecast", forecast))
        await asyncio.gather(*tasks)
        result = evaluate_risk(consensus(list(decisions.values())), risk_evidence, self.limits)
        return AnalysisResult(ticker, agents, decisions, result, errors)
