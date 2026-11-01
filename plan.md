# ADAM — Adaptive Decision Agent Manager

**Your Local AI Hedge Fund Manager**

ADAM is a fully local, open-source AI-powered hedge fund research and decision-making platform that coordinates multiple specialized decision agents to analyze stocks, forecast prices, evaluate financial news, assess fundamentals, and generate consensus-based **Buy, Hold, or Sell** recommendations.

Powered by **Yahoo Finance, Mistral Timestral, FinBERT, D3 via Ollama, and Python**, ADAM performs concurrent multi-stock analysis, streams AI decisions into an interactive dashboard, and applies a mandatory risk gate before issuing final recommendations.

**No paid AI APIs. No cloud inference. No unnecessary infrastructure.**

> ADAM is a research and decision-support system, not an autonomous brokerage or a guarantee of profitable trades.

---

## 1. Project Goals

ADAM aims to provide an efficient, transparent, privacy-first alternative to traditional LLM-based trading agents.

### Core objectives

- **Fully local AI:** All AI inference runs on the user's hardware.
- **Zero paid API costs:** Use open models and freely accessible market data.
- **Fast decisions:** Target sub-second cached decisions and fresh local analysis within 10 seconds.
- **Concurrent execution:** Analyze multiple stocks and independent evidence streams simultaneously.
- **Multiple decision agents:** Generate separate Buy/Hold/Sell assessments for each category.
- **Transparent voting:** Combine agent decisions into a consensus recommendation.
- **Risk-first design:** Apply deterministic risk rules before generating the final recommendation.
- **Live dashboard:** Display each agent's progress, evidence, and decisions as they arrive.
- **Low resource consumption:** Control CPU threads, GPU memory, model loading, and concurrent requests.
- **Simple setup:** Use `uv` to manage dependencies and launch the entire application.

The initial focus is stock analysis and paper-trading research. Portfolio optimization and automated execution may be added later.

---

## 2. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Project management | Python 3.12 + uv | Dependencies and execution |
| Market data | yfinance | Historical prices, volume, fundamentals |
| Forecasting | Mistral Timestral | Probabilistic stock-price forecasts |
| Sentiment | FinBERT | Financial news sentiment |
| Decision AI | D3 | Structured Buy/Hold/Sell assessments |
| AI serving | Ollama | Local D3 inference on Windows, macOS, and Linux |
| News extraction | Trafilatura / Crawl4AI | Extract public financial news |
| Technical analysis | NumPy / pandas | Indicators and market signals |
| Backend | FastAPI | Local application API |
| Dashboard | NiceGUI | Interactive real-time interface |
| Concurrency | asyncio | Parallel task orchestration |
| Sentiment optimization | ONNX Runtime (optional) | Efficient CPU inference |
| Backtesting | vectorbt | Historical strategy evaluation |
| Data caching | Local files / in-memory cache | Avoid unnecessary repeated work |
| Deployment | uv initially | Single-command startup |
| Future deployment | Docker Compose | Optional containerized installation |

Crawl4AI is optional and should only be activated for articles that cannot be extracted through simpler methods.

### Local hardware

The initial implementation targets an NVIDIA RTX 4090 but should support smaller GPUs and CPU-only operation with reduced model configurations.

---

## 3. High-Level Architecture

```text
                     ADAM
          Adaptive Decision Agent Manager
                       |
                User Request
             AAPL, NVDA, TSLA
                       |
              Async Coordinator
                       |
       +---------------+---------------+
       |               |               |
   Market Data     Financial News   Fundamentals
       |               |               |
       |          News Extraction      |
       |               |               |
   +---+---+        FinBERT         Financial
   |       |           |             Metrics
Timestral Technical    |               |
   |       |           |               |
   +-------+-----------+---------------+
                       |
             Shared D3 Decision Engine
                       |
          Independent Category Decisions
                       |
       +---------------+---------------+
       |       |       |       |       |
     Market Forecast News  Technical Fundamentals
       |       |       |       |       |
      BUY     BUY     HOLD    BUY     SELL
       |       |       |       |       |
       +---------------+---------------+
                       |
               Consensus Voting
                       |
                 Risk Gate
                       |
             Final Recommendation
                       |
                Live Dashboard
```

Independent data acquisition and analysis jobs run concurrently. Tasks wait only for the specific inputs they require.

All tickers share the same model instances and bounded inference workers.

---

## 4. Specialized Decision Agents

ADAM uses five core decision agents, all powered by the shared D3 decision model.

These are logical agents, not five separate large language models.

### Agent 1 — Market Analysis

**Data:** Yahoo Finance

Analyzes:

- Market price and volume
- Price momentum
- Historical volatility
- Relative market performance
- Market regime
- Recent price changes

**Output:** Buy / Hold / Sell assessment, model scores, and supporting market metrics.

### Agent 2 — Forecasting

**Model:** Mistral Timestral

Responsibilities:

- Forecast future stock prices.
- Support multiple prediction horizons.
- Calculate median predicted returns.
- Analyze upper and lower forecast quantiles.
- Evaluate forecast uncertainty.
- Compare forecasting performance against historical data and simple baselines.

D3 converts the forecast evidence into a structured Buy/Hold/Sell assessment.

### Agent 3 — Financial News & Sentiment

**Tools:** RSS, Trafilatura, Crawl4AI, FinBERT

Responsibilities:

- Discover relevant financial news.
- Extract clean article content.
- Generate short extractive summaries.
- Identify key financial facts and events.
- Classify positive, neutral, and negative sentiment.
- Identify individual bullish and bearish news events.
- Eliminate duplicate articles.
- Aggregate sentiment across relevant news.

D3 evaluates the combined news evidence to generate a category-level trading signal.

Individual articles may receive their own assessment but do not independently count as votes in the final consensus.

### Agent 4 — Technical Analysis

**Tools:** NumPy and pandas

Indicators include:

- RSI
- MACD
- SMA / EMA
- Bollinger Bands
- ATR
- Momentum
- Volume trends
- Trend strength
- Support and resistance

The agent provides structured indicator values to D3, which produces a technical Buy/Hold/Sell assessment.

### Agent 5 — Fundamental Analysis

**Data:** Yahoo Finance

Analyzes available financial metrics such as:

- Revenue growth
- Earnings growth
- Profitability
- P/E ratio
- Debt and leverage
- Cash flow
- Financial health
- Valuation

Missing or outdated fundamentals are explicitly identified.

D3 produces a fundamental Buy/Hold/Sell assessment based on available evidence.

---

## 5. D3 Decision Engine

D3 is the central structured decision model.

Instead of independently running several generative AI agents, ADAM uses a shared D3 runtime to evaluate category-specific evidence.

Each decision contains:

```json
{
  "ticker": "AAPL",
  "agent": "forecast",
  "signal": "BUY",
  "probabilities": {
    "BUY": 0.70,
    "HOLD": 0.22,
    "SELL": 0.08
  },
  "summary": "Positive median forecast with moderate uncertainty.",
  "status": "completed"
}
```

This represents ADAM's normalized application output, not necessarily the native D3 response.

D3 model scores are not automatically calibrated probabilities of a profitable trade.

### D3 optimization

- Keep one D3 model instance loaded.
- Support multiple concurrent requests.
- Measure Ollama request concurrency and batching behavior.
- Minimize repeated context.
- Use compact structured evidence.
- Compare multiple D3 model sizes for latency and decision quality.
- Avoid unnecessary generative reasoning.

D3 evaluates evidence using predefined decision criteria. Dynamically generated trading rules must be tested before they can affect real recommendations.

---

## 6. Consensus Voting

ADAM aggregates the five category-level decisions.

Example:

| Agent | Decision |
|---|---|
| Market | Buy |
| Forecasting | Buy |
| News | Hold |
| Technical | Buy |
| Fundamentals | Sell |

**Result: BUY (3/5 votes)**

### Voting methods

**Majority voting**

Every available agent receives one vote.

**Weighted voting**

Each agent receives a weight determined through historical out-of-sample testing.

**Adaptive voting — future feature**

Weights or decision thresholds may adjust based on validated performance and changing market conditions.

### Voting rules

- Missing evidence results in abstention.
- Ties default to Hold.
- Unreliable or outdated signals are flagged.
- Agent disagreement must remain visible.
- Voting never bypasses the risk gate.
- Voting confidence is different from actual expected profitability.

---

## 7. Mandatory Risk Gate

The consensus recommendation is not the final decision.

A deterministic Python risk engine must evaluate the recommendation first.

### Risk checks

- Market data freshness
- Volatility thresholds
- Forecast downside uncertainty
- Position size limits
- Portfolio concentration
- Liquidity
- Exposure to correlated assets
- Model disagreement
- Missing critical data
- Risk/reward constraints
- Transaction costs
- Maximum acceptable loss

### Possible outcomes

| Risk result | Action |
|---|---|
| Pass | Approve the consensus recommendation |
| Warning | Flag caution or reduce suggested exposure |
| Fail | Return No Trade |
| Insufficient evidence | Hold / No Trade, depending on position context |

ADAM must clearly distinguish **Hold** from **No Trade**.

The risk engine cannot be overridden by D3 model preferences.

---

## 8. Concurrent Execution Architecture

Concurrency is a central requirement.

### Multiple levels of concurrency

**Ticker-level concurrency**

Multiple stocks are analyzed simultaneously.

**Data-level concurrency**

Independent market, news, and fundamental data collection runs concurrently.

**Analysis-level concurrency**

Forecasting, sentiment, and technical analysis execute as soon as their required inputs are ready.

**Inference-level concurrency**

D3 serves multiple in-flight decisions through a shared inference scheduler.

### Resource control

ADAM must implement:

- Bounded worker pools
- Asynchronous I/O
- Rate-limited data collection
- Model reuse
- Adaptive microbatching
- Request deduplication
- Cancellation and timeouts
- Backpressure
- CPU thread limits
- Configurable inference concurrency

The system should support dozens of tickers in flight without loading duplicate copies of large models.

Concurrency must improve throughput without exhausting RAM, CPU, or VRAM.

---

## 9. Real-Time Dashboard

**Framework:** NiceGUI + FastAPI

The dashboard should display intermediate results immediately as each component finishes.

### Main dashboard

Show:

- Multi-ticker watchlist
- Current market price
- Final recommendation
- Individual agent decisions
- Voting breakdown
- Risk-gate result
- Forecasted returns
- Sentiment overview
- Model agreement
- Evidence freshness
- Analysis progress
- Execution latency

### Stock detail page

Each ticker should have dedicated panels.

**Overview:** Final decision, voting summary, risk status, and evidence summary.

**News Intelligence:** Article headlines, links, key points, summaries, sentiment scores, and individual news signals.

**Forecasting:** Timestral forecast visualization, uncertainty intervals, predicted returns, and historical model performance.

**Technical Analysis:** Indicator values, trend visualization, and the D3 technical assessment.

**Fundamentals:** Growth, valuation, financial metrics, and the fundamental assessment.

**Decision Timeline:** A chronological view of model execution and the arrival of individual decisions.

**Risk Analysis:** Passed, failed, and warning checks.

**Historical Performance:** Previous decisions and their subsequent outcomes.

### Live event updates

Use WebSockets or SSE to stream events such as:

```text
[AAPL] Market data loaded
[NVDA] News collected
[AAPL] Technical indicators ready
[TSLA] Timestral forecast completed
[NVDA] FinBERT sentiment completed
[AAPL] D3 market decision: BUY
[TSLA] D3 forecast decision: HOLD
[AAPL] Consensus: BUY
[AAPL] Risk gate: PASS
[AAPL] Final recommendation: BUY
```

These events are illustrative.

The UI must never wait for all tickers to finish before displaying completed results.

---

## 10. Performance Requirements

ADAM should be optimized for low latency and efficient local inference.

| Metric | Development target |
|---|---|
| Cached decision | Under 1 second |
| Fresh local inference | Under 10 seconds |
| Multi-ticker processing | Concurrent |
| Paid model API cost | $0 |
| Model copies | One per selected model |
| CPU usage | Bounded and configurable |
| Memory | Minimal idle footprint |
| Dashboard | Progressive real-time updates |
| Startup | Automatic model/runtime initialization |

These are engineering targets that require benchmarks on supported hardware. Live external data collection and initial model downloads are outside guaranteed decision-latency targets.

### Optimization strategy

1. Keep the selected D3 model warm during active analysis.
2. Batch related inference requests.
3. Use CPU-efficient Timestral configurations.
4. Run FinBERT only on newly collected content.
5. Avoid unnecessary browser scraping.
6. Cache intermediate analysis outputs.
7. Limit the number of active CPU inference workers.
8. Avoid redundant generative LLM calls.
9. Stream completed evidence immediately.
10. Benchmark latency, memory, and decision quality across model configurations.

Measure p50, p95, and p99 latency, not just average response time.

---

## 11. Installation and Launch

The initial release will use **uv** rather than Docker Compose.

```bash
git clone <repository-url>
cd adam

uv sync --extra forecast --extra sentiment
uv run adam-setup
uv run adam
```

The launcher should automatically:

1. Check the local environment and supported hardware.
2. Locate or download the required model weights during first-time setup.
3. Start the local D3 inference runtime.
4. Load forecasting and sentiment components when required.
5. Initialize the concurrent task coordinator.
6. Start market/news data workers.
7. Launch the dashboard.
8. Display model readiness and runtime health.

The user should never need to manually start several AI servers.

Ollama runs locally on the host operating system. `uv run adam-setup` detects the platform, installs or updates Ollama if necessary, starts its local server, and downloads the configured D3 model.

### Future Docker support

Docker Compose will be introduced after the initial application is stable.

The containerized version should reuse the same application entrypoint.

---

## 12. Backtesting and Evaluation

ADAM must validate whether specialized AI decision models actually improve trading outcomes.

### Benchmark configurations

- Simple rule-based strategy
- Timestral forecast-based signals
- FinBERT sentiment-based signals
- D3 category decisions
- Equal-weight consensus
- Performance-weighted consensus
- Consensus with the risk gate

### Evaluation metrics

- Directional accuracy
- Forecast MAE / RMSE
- Win rate
- Total return
- Sharpe ratio
- Maximum drawdown
- Profit factor
- Transaction costs
- Signal calibration
- Risk-adjusted performance
- Decision latency
- RAM and VRAM consumption

Backtesting must use point-in-time data, chronological validation, and realistic transaction assumptions to avoid look-ahead bias.

Future functionality may include forward testing and paper trading.

---

## 13. Proposed Project Structure

```text
adam/
├── pyproject.toml
├── uv.lock
├── config.yaml
├── README.md
│
├── src/adam/
│   ├── cli.py
│   ├── supervisor.py
│   ├── coordinator.py
│   │
│   ├── data/
│   │   ├── market.py
│   │   ├── news.py
│   │   └── fundamentals.py
│   │
│   ├── models/
│   │   ├── timestral.py
│   │   ├── finbert.py
│   │   └── d3.py
│   │
│   ├── agents/
│   │   ├── market.py
│   │   ├── forecasting.py
│   │   ├── news.py
│   │   ├── technical.py
│   │   └── fundamentals.py
│   │
│   ├── decisions/
│   │   ├── schemas.py
│   │   ├── voting.py
│   │   ├── risk.py
│   │   └── engine.py
│   │
│   ├── runtime/
│   │   ├── batching.py
│   │   ├── workers.py
│   │   └── events.py
│   │
│   ├── evaluation/
│   │   ├── backtest.py
│   │   └── benchmarks.py
│   │
│   └── ui/
│       ├── dashboard.py
│       ├── stocks.py
│       ├── news.py
│       └── decisions.py
│
├── tests/
└── models/
```

---

## 14. Implementation Roadmap

### Phase 1 — Core Application

- [x] Initialize Python project with uv.
- [ ] Set up application configuration.
  - Built-in defaults and optional YAML configure ticker, data, and D3 concurrency; other configuration needs remain under review.
- [ ] Create the startup supervisor.
  - The dashboard reports local Ollama server and D3 model readiness; broader startup and hardware checks remain outstanding.
- [x] Build NiceGUI dashboard shell.
- [x] Implement Yahoo Finance market collection.
- [x] Add multi-ticker requests.
- [x] Implement concurrent execution.

### Phase 2 — Local AI Models

- [x] Integrate Timestral forecasting.
- [x] Integrate FinBERT sentiment.
- [x] Configure D3 with local Ollama.
- [x] Implement structured decision schemas.
- [x] Support multiple concurrent D3 requests.
- [ ] Benchmark inference speed and memory usage.

### Phase 3 — Decision Agents

- [ ] Implement market decision agent.
- [ ] Implement forecasting decision agent.
- [ ] Implement news decision agent.
  - Added source-attributed article extraction from two allowed publishers with robots checks, chunked sentiment, and extractive financial highlights with headline fallback. Broader source coverage, relevance assessment, and event extraction remain outstanding.
- [ ] Implement technical decision agent.
- [ ] Implement fundamental decision agent.
- [x] Add independent category-level assessments.
- [x] Add consensus voting.

### Phase 4 — Risk & Transparency

- [x] Implement mandatory risk gate.
  - The coordinator derives conservative market freshness, volatility, liquidity proxy, and forecast downside; portfolio exposure and trading costs still require supplied evidence.
- [x] Add signal disagreement detection.
- [ ] Show decision scores and explanations.
  - The dashboard shows scores, pre-risk vote counts, selected evidence, forecast quantiles, and news links. D3 does not produce causal explanations; richer evidence presentation remains outstanding.
- [ ] Add article summaries and financial highlights.
- [ ] Implement live event streaming.
- [ ] Add full decision timeline.

### Phase 5 — Performance

- [ ] Add dynamic inference batching.
- [ ] Implement bounded worker pools.
  - Yahoo price and fundamentals calls share a bounded pool and retry selected transient connection failures once; broader workload controls remain outstanding.
- [ ] Optimize CPU thread consumption.
- [ ] Reduce duplicate model loading.
- [ ] Add request deduplication.
  - Concurrent price and fundamentals requests for the same ticker are shared within a coordinator; other request types remain independent.
- [ ] Measure p50/p95/p99 decision latency.
- [ ] Test concurrent workloads across multiple tickers.
  - Added a bounded live coordinator benchmark. On Windows, a 20-ticker run at concurrency 4 completed 18 tickers; two encountered Yahoo connection failures. Successful ticker latency was p50 22.7s, p95 46.3s, p99 51.3s, including network time. Forecast and news coverage and repeat measurements remain outstanding.

### Phase 6 — Evaluation

- [ ] Implement historical backtesting.
- [ ] Compare rule-based and AI-based decisions.
  - Added a trailing moving-average baseline against risk-gated model consensus; an out-of-sample study with recorded historical AI decisions remains outstanding.
- [ ] Evaluate weighted versus majority voting.
  - Implemented comparison for caller-supplied point-in-time decisions and preselected weights; historical dataset and out-of-sample study remain outstanding.
- [ ] Test risk-adjusted performance.
  - Rolling forecast evaluation measures direction and empirical quantile coverage. A calibration command evaluates independently labeled D3 decisions; representative held-out datasets and larger studies remain outstanding.
- [ ] Add decision history and performance tracking.
  - Saved analyses can be evaluated as independent hypothetical Buy outcomes over subsequent completed sessions; portfolio-level tracking remains outstanding.
- [ ] Implement forward and paper-trading evaluation.

### Phase 7 — Distribution

- [ ] Finalize one-command installation.
- [ ] Document GPU and CPU requirements.
  - Added platform and model capacity guidance with links to Ollama's current hardware requirements; additional hardware testing remains outstanding.
- [ ] Add offline operation.
  - Added snapshot-based offline research with cached news, cached Timestral, and mandatory No Trade; validated on Windows with external socket connections blocked. Cross-platform live validation remains outstanding.
- [ ] Publish installation and development guides.
  - Added installation, configuration, and testing guides; full supported-platform installation validation remains outstanding.
- [ ] Add optional Docker Compose packaging.

### Phase 8 — Release

- [ ] Complete the full test suite and verify supported-platform checks.
- [ ] Push verified changes to GitHub.
- [ ] Create the GitHub release with accurate notes.

---

## 15. What Makes ADAM Different?

Unlike many LLM-based trading frameworks that rely on multiple conversational agents debating investment decisions, ADAM is designed around specialized forecasting, financial sentiment, structured AI decisions, and deterministic risk management.

Its key differentiators are:

1. **Specialized local AI models rather than expensive generative LLM debates.**
2. **Concurrent execution across multiple stocks and evidence sources.**
3. **D3-based Buy/Hold/Sell assessments for each category.**
4. **Transparent consensus voting with visible disagreement.**
5. **Mandatory deterministic risk checks before final recommendations.**
6. **Real-time visualization of every stage of the decision process.**
7. **Lightweight uv-first deployment without mandatory cloud infrastructure.**
8. **Performance benchmarking to test both speed and trading effectiveness.**

## 16. Long-Term Vision

ADAM should evolve from a stock research application into a complete, locally operated AI investment decision platform.

Potential future capabilities include portfolio allocation, adaptive risk management, paper trading, broker integrations, strategy optimization, performance-based signal weighting, and automatic evaluation of decisions against realized market outcomes.

The priority is to demonstrate that combining specialized decision models can deliver useful, explainable, low-latency investment research without depending on paid inference APIs.

---

**ADAM — Adaptive Decision Agent Manager**

*Multiple AI Decisions. One Intelligent Strategy.*
