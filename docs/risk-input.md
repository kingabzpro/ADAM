# Risk inputs

ADAM requires explicitly supplied limits and per-ticker evidence before it can approve a research signal. Enter a JSON object in the dashboard. Missing evidence, stale market data, or missing critical market or forecast decisions results in **No Trade**. Limits and evidence are supplied by the user; ADAM does not independently verify the portfolio values or the age of each metric.

The example below shows the format only. Its values are not recommendations, and the timestamp is deliberately stale.

```json
{
  "limits": {
    "max_age_seconds": 86400,
    "max_volatility": 0.5,
    "max_downside": 0.2,
    "max_position_fraction": 0.1,
    "max_concentration": 0.3,
    "min_daily_dollar_volume": 1000000,
    "max_correlated_exposure": 0.4,
    "min_reward_risk": 2,
    "max_transaction_cost_fraction": 0.01,
    "max_loss_fraction": 0.05
  },
  "tickers": {
    "AAPL": {
      "market_time": "2026-01-01T00:00:00Z",
      "volatility": 0.2,
      "downside": 0.1,
      "position_fraction": 0.05,
      "concentration": 0.2,
      "daily_dollar_volume": 2000000,
      "correlated_exposure": 0.3,
      "reward_risk": 3,
      "transaction_cost_fraction": 0.005,
      "loss_fraction": 0.02
    }
  }
}
```

`market_time` must include a timezone. The gate uses the earlier of this value and the timestamp of the latest daily price bar, which may conservatively predate the market close. Exposure, downside, costs, and loss are fractions; volatility must use the same units as its limit. Liquidity is daily dollar volume and reward/risk is a ratio. Each ticker needs its own evidence. **Hold** and **No Trade** are distinct results.
