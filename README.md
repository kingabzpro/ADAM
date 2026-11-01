# Adaptive Decision Agent Manager (ADAM)
ADAM is a fully local, open-source AI hedge fund research platform that coordinates specialized decision agents for market analysis, price forecasting, news sentiment, and fundamentals. Powered by **Yahoo Finance, Mistral Timestral, FinBERT, and D3 via vLLM**, it generates consensus-based Buy/Hold/Sell signals with real-time analysis and built-in risk management—all without paid AI APIs.

Configuration currently defines `inference_concurrency` in `config.yaml`. The inference runtime will consume it when implemented.

Run `uv sync` followed by `uv run adam` from the project directory. Enter comma-separated tickers and select Load prices to see the latest available adjusted daily closes.

The decision and risk modules are not yet connected to live analysis. The risk gate requires explicitly supplied limits and evidence: exposure, downside, loss and cost values are fractions; volatility is a non-negative measure in the same units as its limit; liquidity is daily dollar volume; reward/risk is a ratio; and market timestamps must include a timezone. Missing evidence returns No Trade. No default trading limits are assumed.

D3 integration targets vLLM's `/v1/systemone` endpoint and accepts only a loopback HTTP origin. It expects a separately started, compatible D3 server and an `httpx.AsyncClient` configured without a proxy (`trust_env=False`). D3 produces a classification and scores, not a generated explanation or a calibrated probability of profit. The dashboard does not yet invoke this client.
