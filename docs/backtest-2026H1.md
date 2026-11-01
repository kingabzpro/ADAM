# ADAM weekly backtest: 2026-01-01 to 2026-06-30

Universe: 25 tickers (AAPL, MSFT, NVDA, GOOGL, AMZN, META, BRK-B, TSLA, LLY, AVGO, JPM, V, XOM, UNH, MA, PG, JNJ, COST, HD, WMT, ORCL, MRK, ABBV, CVX, CRM). Weekly decisions from the four-agent consensus (market, technical, forecast, fundamentals; no historical news), executed long/flat at the next session's open, transaction cost 5.0 bps per trade.

## Aggregate (equal weight)

| Strategy | Total return | Sharpe | Max drawdown |
|---|---|---|---|
| ADAM weekly consensus | -0.8% | -0.19 | +5.0% |
| Buy and hold | +13.6% | 1.50 | +7.4% |
| SMA-20 rule | -1.5% | -0.35 | +3.8% |

## Signal quality (pooled over all ticker-weeks)

| Signal | Weeks | Mean next-week return | Directional accuracy |
|---|---|---|---|
| BUY | 268 | -0.1% | +53.7% |
| SELL | 241 | +0.7% | +46.5% |
| HOLD | 116 | n/a | n/a |

## Per ticker

| Ticker | Strategy | Buy and hold | SMA rule | BUY accuracy | SELL accuracy | Errors |
|---|---|---|---|---|---|---|
| AAPL | +7.5% | +36.8% | +13.2% | +61.5% | +57.1% | 0 |
| MSFT | -10.7% | -22.7% | -3.3% | +50.0% | +54.5% | 0 |
| NVDA | +6.2% | +14.4% | +0.2% | +47.6% | n/a | 0 |
| GOOGL | -1.3% | +67.8% | +22.2% | +42.9% | +42.9% | 0 |
| AMZN | -5.0% | +6.9% | -12.7% | +69.2% | +55.6% | 0 |
| META | -21.6% | -11.6% | -13.1% | +14.3% | +40.0% | 0 |
| BRK-B | -2.6% | -0.8% | -1.7% | +50.0% | +30.0% | 0 |
| TSLA | +0.0% | +13.6% | -7.0% | n/a | +60.0% | 0 |
| LLY | +6.4% | +62.1% | -7.9% | +50.0% | +40.0% | 0 |
| AVGO | -6.3% | +25.9% | -16.1% | +38.5% | +20.0% | 0 |
| JPM | -10.4% | +12.8% | -10.1% | +50.0% | +38.5% | 0 |
| V | +13.2% | +2.7% | +2.5% | +37.5% | +54.5% | 0 |
| XOM | +9.5% | +29.9% | +10.5% | +60.0% | +33.3% | 0 |
| UNH | +7.4% | +45.7% | +14.7% | +63.6% | +50.0% | 0 |
| MA | -0.5% | -8.7% | -4.0% | +0.0% | +47.6% | 0 |
| PG | -0.5% | -2.9% | +6.2% | +66.7% | +38.5% | 0 |
| JNJ | +14.5% | +49.0% | +4.1% | +63.6% | +44.4% | 0 |
| COST | +3.9% | -1.6% | +0.8% | +57.9% | +20.0% | 0 |
| HD | -6.1% | -15.2% | +4.9% | +42.9% | +46.2% | 0 |
| WMT | +7.3% | +20.1% | -7.0% | +68.8% | +60.0% | 0 |
| ORCL | -12.6% | -44.0% | +0.6% | +37.5% | +62.5% | 0 |
| MRK | +1.8% | +53.6% | -12.3% | +46.7% | +20.0% | 0 |
| ABBV | -3.1% | +22.0% | -7.9% | +40.0% | +35.7% | 0 |
| CVX | +8.3% | +17.5% | +6.6% | +75.0% | +50.0% | 0 |
| CRM | -26.3% | -32.4% | -20.3% | +33.3% | +53.3% | 0 |

## Margin sweep (one pipeline pass, consensus recomputed per margin)

| Margin | Strategy return | BUY weeks | SELL weeks | HOLD weeks | BUY accuracy | SELL accuracy |
|---|---|---|---|---|---|---|
| 0 | -1.5% | 326 | 299 | 0 | +52.5% | +48.2% |
| 0.025 | -1.9% | 308 | 286 | 31 | +52.6% | +48.6% |
| 0.05 | -1.7% | 289 | 267 | 69 | +52.9% | +48.3% |
| 0.075 | -0.8% | 268 | 241 | 116 | +53.7% | +46.5% |
| 0.1 | -1.4% | 254 | 216 | 155 | +55.1% | +46.3% |
| 0.15 | -2.7% | 231 | 168 | 226 | +54.5% | +45.2% |
| 0.2 | -1.0% | 195 | 124 | 306 | +55.4% | +44.4% |

## Caveats

- Fundamentals are today's data applied to historical weeks (Yahoo has no point-in-time fundamentals), so the fundamentals agent carries lookahead bias.
- No historical news is replayed; votes come from four agents.
- The universe is today's largest companies, which biases the study toward survivors.
- Weekly closes and opens come from Yahoo-adjusted daily bars; signals decide at a week's last session and execute at the next session's open.
- Decision quality does not establish profitability; this is a research measurement.
