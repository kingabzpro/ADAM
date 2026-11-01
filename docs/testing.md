# Testing

Run the offline unit suite from the project root:

```bash
uv run python -m unittest discover -s tests
```

The suite mocks model and external network calls and includes a local loopback HTTP retry test; it does not install Ollama or download weights. To verify real D3 inference, first run `uv run adam-setup`, then enable the opt-in local test.

**macOS / Linux**

```bash
ADAM_TEST_LIVE_OLLAMA=1 uv run python -m unittest discover -s tests -p test_live_ollama.py
```

**PowerShell**

```powershell
$env:ADAM_TEST_LIVE_OLLAMA = '1'
uv run python -m unittest discover -s tests -p test_live_ollama.py
Remove-Item Env:ADAM_TEST_LIVE_OLLAMA
```

This check requires the local Ollama server and `d3:2b` model. It runs inference and a small concurrent benchmark; it does not establish model quality or trading performance. `uv run adam-benchmark --help` describes the standalone benchmark options.

Run a weekly historical backtest with `uv run adam-backtest` (defaults: top 25 S&P mega caps, 2026-01-01 to 2026-06-30, weekly decisions, long/flat execution at the next open, 5 bps per trade). It fetches each ticker's history once, slices point-in-time windows, and writes a markdown report with `--out`. Fundamentals are current-day data and no news is replayed, so votes come from four agents; the report states the caveats.

Run a live multi-ticker coordinator benchmark with `uv run adam-benchmark --tickers AAPL,MSFT,NVDA,AMZN --concurrency 2`. It fetches current Yahoo data and runs the market, technical, and fundamentals agents through D3, with one D3 request at a time. Forecast and news are disabled; risk inputs are absent, so results are No Trade. It reports successful per-ticker latency percentiles, throughput, agent status, and failures. Times include network access and are not model-only measurements. Use more tickers and repeated runs for meaningful tail latency; this is not a quality or trading-performance test.

Each coordinator allows four concurrent Yahoo price or fundamentals calls by default. Prices and fundamentals load in one combined fetch per ticker, and concurrent requests for the same ticker share one in-flight call. Online analysis resolves prices, fundamentals, and news through a process-wide TTL cache controlled by `cache_ttl_seconds` (900 seconds by default): a repeat analysis inside the window reuses the cached decision and completes without network or inference, and after a restart, snapshots saved within the window are served before any fetch. The risk gate is always recomputed from the current limits and risk inputs, even on a cache hit. With the cache disabled or expired, completed calls are not reused. Separate local snapshots support offline research. Cancelling a waiting analysis does not stop a shared call already running in a thread.

Yahoo price and fundamentals requests retry once after a short randomized delay for selected connection and timeout failures. Rate limits, certificate errors, invalid requests, and other failures are not retried. A failed retry is reported to the relevant agent and cannot supply fresh evidence.

`adam.evaluation.evaluate_forecast` evaluates a forecaster over expanding historical windows. It reports model and last-price baseline MAE/RMSE, direction accuracy, and empirical 10th/90th quantile and interval coverage. Direction accuracy excludes unchanged actual prices; a flat forecast counts as incorrect when the price moves. Coverage counts observations on interval boundaries as inside, allowing for floating-point reconstruction error. `step` and `horizon` determine the sample spacing; overlapping horizons are not independent. Provide point-in-time, consistently adjusted prices and account for the model's training period: the function cannot verify either condition, and small samples do not establish calibration or profitability.

See [decision-score calibration](calibration.md) for evaluating D3 choice scores against independently labeled examples.

For historical research, `adam.evaluation.compare_voting` compares majority and weighted votes using the same risk limits, transaction costs, and next-session execution. Supply one set of agent decisions and `RiskEvidence` for each timezone-aware price index timestamp, plus weights fixed before the evaluation period. The index timestamps must represent the decision times. Market and forecast decisions must both be available and reliable; otherwise the result is No Trade. The caller must ensure that every decision, input, price adjustment, and weight was available at that time: the comparison cannot detect look-ahead bias. It does not fit weights or recreate historical AI decisions.

The comparison invests at most the supplied `position_fraction` of available cash on a Buy, including fees; it does not rebalance an open position. A Sell closes the position. The standalone `backtest` defaults to investing all available cash unless `position_fractions` is supplied. Neither simulates a multi-asset portfolio, slippage, taxes, or corporate actions.

`adam.evaluation.compare_rule_baseline` compares risk-gated model consensus against a simple trailing moving-average rule over the same sessions. The rule buys when the close is above the average, sells below it, and holds at equality. Both strategies wait for the same warm-up, use the same risk evidence and position fraction, and execute on the following session. The model path additionally requires reliable market and forecast decisions. This is a research baseline, not evidence that either strategy is profitable.

`uv run adam-outcomes TICKER` evaluates up to 1,000 saved, risk-approved Buy analyses for one ticker against one year of Yahoo daily prices. Each hypothetical outcome enters at the open of the first later trading date and exits at the fifth completed session's close. Today's session is excluded; older decisions outside the downloaded price history are marked unavailable, and incomplete horizons remain pending. The report assumes zero transaction costs by default and uses adjusted prices. Outcomes can overlap and are independent observations, not actual fills, a portfolio return, or proof of future results. Missing sessions in the source data can distort results.
