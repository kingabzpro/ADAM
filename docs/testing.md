# Testing

Run the offline unit suite from the project root:

```bash
uv run python -m unittest discover -s tests
```

The suite uses mocked model and network calls; it does not install Ollama or download weights. To verify real D3 inference, first run `uv run adam-setup`, then enable the opt-in local test.

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

Run a live multi-ticker coordinator benchmark with `uv run adam-benchmark --tickers AAPL,MSFT,NVDA,AMZN --concurrency 2`. It fetches current Yahoo data and runs the market, technical, and fundamentals agents through D3, with one D3 request at a time. Forecast and news are disabled; risk inputs are absent, so results are No Trade. It reports successful per-ticker latency percentiles, throughput, agent status, and failures. Times include network access and are not model-only measurements. Use more tickers and repeated runs for meaningful tail latency; this is not a quality or trading-performance test.

Each coordinator allows four concurrent Yahoo price or fundamentals calls by default. Concurrent requests for the same ticker and data type share one in-flight call; online analysis does not reuse completed calls except for the short-lived SPY benchmark cache. Separate local snapshots support offline research. Cancelling a waiting analysis does not stop a shared call already running in a thread.

`adam.evaluation.evaluate_forecast` evaluates a forecaster over expanding historical windows. It reports model and last-price baseline MAE/RMSE, direction accuracy, and empirical 10th/90th quantile and interval coverage. Direction accuracy excludes unchanged actual prices; a flat forecast counts as incorrect when the price moves. Coverage counts observations on interval boundaries as inside, allowing for floating-point reconstruction error. `step` and `horizon` determine the sample spacing; overlapping horizons are not independent. Provide point-in-time, consistently adjusted prices and account for the model's training period: the function cannot verify either condition, and small samples do not establish calibration or profitability.

See [decision-score calibration](calibration.md) for evaluating D3 choice scores against independently labeled examples.

For historical research, `adam.evaluation.compare_voting` compares majority and weighted votes using the same risk limits, transaction costs, and next-session execution. Supply one set of agent decisions and `RiskEvidence` for each timezone-aware price index timestamp, plus weights fixed before the evaluation period. The index timestamps must represent the decision times. Market and forecast decisions must both be available and reliable; otherwise the result is No Trade. The caller must ensure that every decision, input, price adjustment, and weight was available at that time: the comparison cannot detect look-ahead bias. It does not fit weights or recreate historical AI decisions.

The comparison invests at most the supplied `position_fraction` of available cash on a Buy, including fees; it does not rebalance an open position. A Sell closes the position. The standalone `backtest` defaults to investing all available cash unless `position_fractions` is supplied. Neither simulates a multi-asset portfolio, slippage, taxes, or corporate actions.

`adam.evaluation.compare_rule_baseline` compares risk-gated model consensus against a simple trailing moving-average rule over the same sessions. The rule buys when the close is above the average, sells below it, and holds at equality. Both strategies wait for the same warm-up, use the same risk evidence and position fraction, and execute on the following session. The model path additionally requires reliable market and forecast decisions. This is a research baseline, not evidence that either strategy is profitable.

The dashboard can evaluate up to 1,000 saved, risk-approved Buy analyses for one ticker against one year of Yahoo daily prices. Each hypothetical outcome enters at the open of the first later trading date and exits at the fifth completed session's close. Today's session is excluded; older decisions outside the downloaded price history are marked unavailable, and incomplete horizons remain pending. The display assumes zero transaction costs and uses adjusted prices. Outcomes can overlap and are independent observations, not actual fills, a portfolio return, or proof of future results. Missing sessions in the source data can distort results.
