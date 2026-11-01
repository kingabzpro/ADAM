# Adaptive Decision Agent Manager (ADAM)
ADAM is a fully local, open-source AI hedge fund research platform that coordinates specialized decision agents for market analysis, price forecasting, news sentiment, and fundamentals. Powered by **Yahoo Finance, Mistral Timestral, FinBERT, and D3 via vLLM**, it generates consensus-based Buy/Hold/Sell signals with real-time analysis and built-in risk management—all without paid AI APIs.

Configuration currently defines `inference_concurrency` in `config.yaml`. The inference runtime will consume it when implemented.

Run `uv sync` followed by `uv run adam` from the project directory. Enter comma-separated tickers and select Load prices to see the latest available adjusted daily closes.

The decision and risk modules are not yet connected to live analysis. The risk gate requires explicitly supplied limits and evidence: exposure, downside, loss and cost values are fractions; volatility is a non-negative measure in the same units as its limit; liquidity is daily dollar volume; reward/risk is a ratio; and market timestamps must include a timezone. Missing evidence returns No Trade. No default trading limits are assumed.

D3 integration targets vLLM's `/v1/systemone` endpoint and accepts only a loopback HTTP origin. It expects a separately started, compatible D3 server and an `httpx.AsyncClient` configured without a proxy (`trust_env=False`). D3 produces a classification and scores, not a generated explanation or a calibrated probability of profit. The dashboard does not yet invoke this client.

Timestral forecasting is an optional local component: run `uv sync --extra forecast` to install it. `adam.forecast.load_forecaster()` downloads the 5M model on first use and runs on CPU by default; pass `device="cuda"` with a compatible PyTorch installation or `local_files_only=True` to use cached weights offline. Forecast horizons are numbers of observations, and output includes 10th, 50th and 90th percentile returns. The dashboard does not yet run forecasts.

FinBERT sentiment is optional: install with `uv sync --extra sentiment`. `adam.sentiment.load_classifier()` loads `ProsusAI/finbert` on CPU and downloads its weights on first use; `local_files_only=True` uses cached weights. Texts longer than the model limit are truncated. The published model weights are licensed CC BY-NC-SA 4.0; review the [model card](https://huggingface.co/ProsusAI/finbert) before use. News collection and dashboard integration are not yet implemented.
