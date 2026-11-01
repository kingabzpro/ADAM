# Risk inputs

ADAM requires explicitly supplied limits and per-ticker portfolio evidence before it can approve a research signal. The web page uses conservative built-in limits; to approve trades, supply your own `RiskLimits` and per-ticker `RiskEvidence` programmatically to `adam.dashboard.run_analysis`, or parse the JSON format below with `adam.dashboard.parse_risk_input`. Missing evidence, stale market data, or missing critical market or forecast decisions results in **No Trade**. ADAM derives some market inputs as described below; it does not independently verify the supplied portfolio values.

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

ADAM derives the latest price-bar timestamp, annualized daily-return volatility, and a liquidity proxy (latest close multiplied by reported daily volume) from market data. It derives downside from the worst lower-quantile forecast return across the horizon, capped at 100%. If corresponding values are also supplied, the gate uses the earlier timestamp, higher volatility and downside, and lower liquidity. Invalid observed values remain missing. The liquidity figure is an estimate, not a verified execution price or guarantee of liquidity; adjusted prices and volume can affect it. The forecast quantile is model output, not a guaranteed loss bound. The dashboard and saved history show both supplied and evaluated risk inputs.

If supplied, `market_time` must include a timezone; daily bar timestamps may conservatively predate the market close. Exposure, downside, costs, and loss are fractions; volatility and its limit use annualized units. Liquidity is estimated daily dollar volume and reward/risk is a ratio. Each ticker needs its own portfolio evidence. **Hold** and **No Trade** are distinct results.
