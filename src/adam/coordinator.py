import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

from adam import cache
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
    def __init__(self, model, limits: RiskLimits | None, *, forecaster=None, news_provider=None, inference_concurrency: int = 1, data_concurrency: int = 4, offline: bool = False, snapshot_directory=None, cache_ttl_seconds: int = 0):
        if type(inference_concurrency) is not int or inference_concurrency < 1:
            raise ValueError("inference_concurrency must be a positive integer")
        if type(data_concurrency) is not int or data_concurrency < 1:
            raise ValueError("data_concurrency must be a positive integer")
        if type(cache_ttl_seconds) is not int or cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be a non-negative integer")
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
        self.cache_ttl_seconds = cache_ttl_seconds
        self.inference_slots = asyncio.Semaphore(inference_concurrency)
        self.forecast_slot = asyncio.Semaphore(1)
        self.data_slots = asyncio.Semaphore(data_concurrency)
        self._data_tasks = {}

    def _fresh_snapshot(self, reader, ticker, ttl, kind):
        """Return (data, version) from a snapshot saved within ttl, or (None, None)."""
        if self.snapshot_directory is None or ttl <= 0:
            return None, None
        try:
            data, saved = reader(self.snapshot_directory, ticker)
        except (OSError, ValueError, KeyError, TypeError):
            return None, None
        if (datetime.now(timezone.utc) - saved).total_seconds() > ttl:
            return None, None
        return data, cache.store(kind, ticker, data, ttl)

    def _save_snapshot(self, saver, ticker, kind, data):
        if self.snapshot_directory is None:
            return
        try:
            saver(self.snapshot_directory, ticker, data)
            self.snapshot_errors.pop((kind, ticker), None)
        except (OSError, ValueError, TypeError) as error:
            self.snapshot_errors[(kind, ticker)] = str(error)
            logging.exception('Unable to save %s snapshot for %s', kind, ticker)

    def _load_data(self, ticker, *, want_info=True):
        """Load 3-month prices and fundamentals from the cache, a fresh snapshot, or one combined fetch."""
        from adam.snapshots import read_fundamentals, read_prices, save_fundamentals, save_prices
        if self.offline:
            try:
                history, saved = read_prices(self.snapshot_directory, ticker)
            except Exception as error:
                history = error
            else:
                self.snapshot_times[('prices', ticker)] = saved.isoformat()
            info = None
            if want_info:
                try:
                    info, saved = read_fundamentals(self.snapshot_directory, ticker)
                except Exception as error:
                    info = error
                else:
                    self.snapshot_times[('fundamentals', ticker)] = saved.isoformat()
            return history, info, (None, None)
        ttl = self.cache_ttl_seconds
        cached_prices = cache.cached('prices', ticker, ttl)
        history, prices_version = cached_prices if cached_prices is not None else (None, None)
        cached_info = cache.cached('fundamentals', ticker, ttl) if want_info else None
        info, info_version = cached_info if cached_info is not None else (None, None)
        if history is None:
            history, prices_version = self._fresh_snapshot(read_prices, ticker, ttl, 'prices')
        if want_info and info is None:
            info, info_version = self._fresh_snapshot(read_fundamentals, ticker, ttl, 'fundamentals')
        if history is not None:
            # Served without a fetch, so no save is attempted and an older save failure is not this run's error.
            self.snapshot_errors.pop(('prices', ticker), None)
        if want_info and info is not None:
            self.snapshot_errors.pop(('fundamentals', ticker), None)
        if history is None or (want_info and info is None):
            if history is None:
                try:
                    history = fetch_history(ticker, '3mo')
                except Exception as error:
                    history = error
            if want_info and info is None:
                try:
                    info = fetch_fundamentals(ticker)
                except Exception as error:
                    info = error
            if not isinstance(history, Exception):
                self._save_snapshot(save_prices, ticker, 'prices', history)
                prices_version = cache.store('prices', ticker, history, ttl)
            if want_info and not isinstance(info, Exception):
                self._save_snapshot(save_fundamentals, ticker, 'fundamentals', info)
                info_version = cache.store('fundamentals', ticker, info, ttl)
        return history, info, (prices_version, info_version)

    async def _load_news(self, ticker):
        """Resolve news evidence from the cache, a fresh snapshot, or the provider."""
        ttl = self.cache_ttl_seconds
        if self.offline or ttl <= 0:
            try:
                return await self.news_provider(ticker), False, None
            except Exception as error:
                return error, False, None
        cached_news = cache.cached('news', ticker, ttl)
        if cached_news is not None:
            return cached_news[0], True, cached_news[1]
        if self.snapshot_directory is not None:
            from adam.snapshots import read_news
            try:
                evidence, saved = await asyncio.to_thread(read_news, self.snapshot_directory, ticker)
            except (OSError, ValueError, KeyError, TypeError):
                pass
            else:
                if (datetime.now(timezone.utc) - saved).total_seconds() <= ttl:
                    marked = {**evidence, 'source': 'local snapshot', 'snapshot_saved_at': saved.isoformat()}
                    return marked, True, cache.store('news', ticker, marked, ttl)
        try:
            evidence = await self.news_provider(ticker)
        except Exception as error:
            return error, False, None
        return evidence, False, cache.store('news', ticker, evidence, ttl)

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
        history, _, _ = await self._fetch_data(('prices', 'SPY'), lambda: self._load_data('SPY', want_info=False))
        if isinstance(history, Exception):
            raise history
        return history

    async def analyze(self, ticker: str, risk_evidence: RiskEvidence, *, horizon: int = 5, on_event=None) -> AnalysisResult:
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker is required")
        agents = {name: "unavailable" for name in ("market", "technical", "forecast", "news", "fundamentals")}
        decisions = {}
        errors = {}
        timeline = []
        evidence_by_agent = {}

        def emit(stage, status, error=None):
            event = {'time': datetime.now(timezone.utc).isoformat(), 'ticker': ticker,
                     'stage': stage, 'status': status}
            if error is not None:
                event['error'] = str(error)
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
                emit(name, "failed", error)

        emit("market_data", "started")
        data_task = asyncio.create_task(self._fetch_data(('data', ticker), self._load_data, ticker))
        benchmark_task = (None if ticker == 'SPY'
                          else asyncio.create_task(self._fetch_data(('prices', 'SPY'), lambda: self._load_data('SPY', want_info=False))))
        news_task = asyncio.create_task(self._load_news(ticker)) if self.news_provider is not None else None
        try:
            history, info, versions = await data_task
        except Exception as error:
            history, info, versions = error, error, (None, None)
        benchmark, benchmark_versions = None, (None, None)
        if benchmark_task is not None:
            try:
                benchmark, _, benchmark_versions = await benchmark_task
            except Exception as error:
                benchmark, benchmark_versions = error, (None, None)
        news_loaded, news_from_cache, news_version = None, False, None
        if news_task is not None:
            try:
                news_loaded, news_from_cache, news_version = await news_task
            except Exception as error:
                news_loaded, news_from_cache, news_version = error, False, None
        if isinstance(history, Exception):
            errors["market_data"] = str(history)
            emit("market_data", "failed", history)
            agents["market"] = agents["technical"] = "failed"
            if self.forecaster is not None:
                agents["forecast"] = "failed"
            for stage in ("benchmark", "market", "technical") + (("forecast",) if self.forecaster is not None else ()):
                emit(stage, "skipped")
        else:
            emit("market_data", "completed")

        def decision_key():
            spy_version = versions[0] if ticker == 'SPY' else benchmark_versions[0]
            return (ticker, versions[0], spy_version, versions[1],
                    'off' if self.news_provider is None else news_version,
                    self.forecaster is not None, horizon)

        def finalize():
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
            return evaluated_risk, result

        cached_decision = (None if self.offline
                           else cache.cached('decision', decision_key(), self.cache_ttl_seconds))
        if cached_decision is not None:
            payload = cached_decision[0]
            agents.update(payload['agents'])
            decisions.update(payload['decisions'])
            errors.update(payload['errors'])
            evidence_by_agent.update(payload['evidence'])
            emit("cache", "hit")
            evaluated_risk, result = finalize()
            emit("risk", result.action)
            return AnalysisResult(ticker, agents, decisions, result, errors, timeline, evidence_by_agent, evaluated_risk)

        async def fundamentals():
            if isinstance(info, Exception):
                raise info
            evidence = fundamentals_evidence(info)
            if self.offline:
                evidence.update(source='local snapshot', snapshot_saved_at=self.snapshot_times[('fundamentals', ticker)])
            if not evidence["metrics"] or evidence["freshness"] != "current":
                raise ValueError("Fundamentals missing or fiscal quarter date stale or unknown")
            return evidence

        async def market():
            evidence = await asyncio.to_thread(market_evidence, history)
            if self.offline:
                evidence.update(source='local snapshot', snapshot_saved_at=self.snapshot_times[('prices', ticker)])
            emit('benchmark', 'started')
            try:
                comparison = history if ticker == 'SPY' else benchmark
                if isinstance(comparison, Exception):
                    raise comparison
                evidence['comparison'] = await asyncio.to_thread(compare_market, history, comparison)
                evidence['comparison_status'] = 'available'
                if self.offline:
                    evidence['benchmark_snapshot_saved_at'] = self.snapshot_times[('prices', 'SPY')]
                emit('benchmark', 'completed')
            except Exception as error:
                evidence['comparison'] = None
                evidence['comparison_status'] = 'unavailable'
                errors['benchmark'] = str(error)
                emit('benchmark', 'failed', error)
            return evidence

        async def market_stream():
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
            if isinstance(news_loaded, Exception):
                raise news_loaded
            evidence = news_loaded
            if not (evidence.get('headlines') or evidence.get('articles')) or evidence.get('mean_scores') is None:
                raise ValueError('No recent news available')
            if not self.offline and not news_from_cache and self.snapshot_directory is not None:
                from adam.snapshots import save_news
                try:
                    await asyncio.to_thread(save_news, self.snapshot_directory, ticker, evidence)
                except (OSError, ValueError, TypeError) as error:
                    errors['snapshot_news'] = str(error)
            return evidence

        tasks = []
        if not isinstance(history, Exception):
            tasks.append(market_stream())
        tasks.append(run("fundamentals", fundamentals))
        if self.news_provider is not None:
            tasks.append(run('news', news))
        await asyncio.gather(*tasks)
        for kind, symbol in (('prices', ticker), ('fundamentals', ticker), ('prices', 'SPY')):
            if (kind, symbol) in self.snapshot_errors:
                label = f'snapshot_{kind}' if symbol == ticker else 'snapshot_benchmark'
                errors[label] = self.snapshot_errors[(kind, symbol)]
        evaluated_risk, result = finalize()
        emit("risk", result.action)
        versions_complete = (versions[0] is not None and versions[1] is not None
                             and (ticker == 'SPY' or benchmark_versions[0] is not None)
                             and (self.news_provider is None or news_version is not None))
        if not self.offline and not errors and versions_complete:
            cache.store('decision', decision_key(),
                        {'agents': dict(agents), 'decisions': dict(decisions),
                         'errors': dict(errors), 'evidence': dict(evidence_by_agent)},
                        self.cache_ttl_seconds)
        return AnalysisResult(ticker, agents, decisions, result, errors, timeline, evidence_by_agent, evaluated_risk)
