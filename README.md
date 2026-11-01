# Adaptive Decision Agent Manager (ADAM)
ADAM is a fully local, open-source AI hedge fund research platform that coordinates specialized decision agents for market analysis, price forecasting, news sentiment, and fundamentals. Powered by **Yahoo Finance, Mistral Timestral, FinBERT, and D3 via vLLM**, it generates consensus-based Buy/Hold/Sell signals with real-time analysis and built-in risk management—all without paid AI APIs.

Configuration currently defines `inference_concurrency` in `config.yaml`. The inference runtime will consume it when implemented.

Run `uv sync` followed by `uv run adam` from the project directory. Enter comma-separated tickers and select Load prices to see the latest available adjusted daily closes.

The decision and risk modules are not yet connected to live analysis. The risk gate requires explicitly supplied limits and evidence: exposure, downside, loss and cost values are fractions; volatility is a non-negative measure in the same units as its limit; liquidity is daily dollar volume; reward/risk is a ratio; and market timestamps must include a timezone. Missing evidence returns No Trade. No default trading limits are assumed.

D3 integration targets vLLM's `/v1/systemone` endpoint and accepts only a loopback HTTP origin. It expects a separately started, compatible D3 server and an `httpx.AsyncClient` configured without a proxy (`trust_env=False`). D3 produces a classification and scores, not a generated explanation or a calibrated probability of profit. The dashboard does not yet invoke this client.

Timestral forecasting is an optional local component: run `uv sync --extra forecast` to install it. `adam.forecast.load_forecaster()` downloads the 5M model on first use and runs on CPU by default; pass `device="cuda"` with a compatible PyTorch installation or `local_files_only=True` to use cached weights offline. Forecast horizons are numbers of observations, and output includes 10th, 50th and 90th percentile returns. The dashboard does not yet run forecasts.

FinBERT sentiment is optional: install with `uv sync --extra sentiment`. `adam.sentiment.load_classifier()` loads `ProsusAI/finbert` on CPU and downloads its weights on first use; `local_files_only=True` uses cached weights. Texts longer than the model limit are truncated. The published model weights are licensed CC BY-NC-SA 4.0; review the [model card](https://huggingface.co/ProsusAI/finbert) before use. News collection and dashboard integration are not yet implemented.

The dashboard's optional risk JSON takes `limits` and `tickers` objects. Each ticker needs its own evidence; omitted fields cause No Trade. The following values illustrate the format only and are not recommended limits or current market data:

```json
{
  "limits": {
    "max_age_seconds": 86400, "max_volatility": 0.5, "max_downside": 0.2,
    "max_position_fraction": 0.1, "max_concentration": 0.3,
    "min_daily_dollar_volume": 1000000, "max_correlated_exposure": 0.4,
    "min_reward_risk": 2, "max_transaction_cost_fraction": 0.01, "max_loss_fraction": 0.05
  },
  "tickers": {
    "AAPL": {
      "market_time": "2026-01-01T00:00:00Z", "volatility": 0.2, "downside": 0.1,
      "position_fraction": 0.05, "concentration": 0.2, "daily_dollar_volume": 2000000,
      "correlated_exposure": 0.3, "reward_risk": 3, "transaction_cost_fraction": 0.005,
      "loss_fraction": 0.02
    }
  }
}
```

Risk evidence is currently supplied manually; the application does not yet calculate portfolio exposures or establish the freshness of each supplied metric. The example timestamp will be stale and therefore cannot approve a trade.
