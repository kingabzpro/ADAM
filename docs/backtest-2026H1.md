# ADAM weekly backtest: 2026-01-01 to 2026-06-30

Universe: 25 tickers (AAPL, MSFT, NVDA, GOOGL, AMZN, META, BRK-B, TSLA, LLY, AVGO, JPM, V, XOM, UNH, MA, PG, JNJ, COST, HD, WMT, ORCL, MRK, ABBV, CVX, CRM). Weekly decisions from the four-agent consensus (market, technical, forecast, fundamentals; no historical news), executed long/flat at the next session's open, transaction cost 5.0 bps per trade.

## Aggregate (equal weight)

| Strategy | Total return | Sharpe | Max drawdown |
|---|---|---|---|
| ADAM weekly consensus | +2.0% | 0.96 | +1.8% |
| Buy and hold | +13.6% | 1.50 | +7.4% |
| SMA-20 rule | -1.5% | -0.35 | +3.8% |

## Signal quality (pooled over all ticker-weeks)

| Signal | Weeks | Mean next-week return | Directional accuracy |
|---|---|---|---|
| BUY | 50 | +0.5% | +64.0% |
| SELL | 66 | +1.2% | +42.4% |
| HOLD | 509 | n/a | n/a |

## Per ticker

| Ticker | Strategy | Buy and hold | SMA rule | BUY accuracy | SELL accuracy | Errors |
|---|---|---|---|---|---|---|
| AAPL | +17.4% | +36.8% | +13.2% | +83.3% | +40.0% | 0 |
| MSFT | -10.7% | -22.7% | -3.3% | +33.3% | +54.5% | 0 |
| NVDA | -3.1% | +14.4% | +0.2% | +40.0% | +0.0% | 0 |
| GOOGL | +0.0% | +67.8% | +22.2% | n/a | n/a | 0 |
| AMZN | +0.0% | +6.9% | -12.7% | n/a | n/a | 0 |
| META | -21.1% | -11.6% | -13.1% | +0.0% | +44.4% | 0 |
| BRK-B | -0.2% | -0.8% | -1.7% | +50.0% | +25.0% | 0 |
| TSLA | +0.0% | +13.6% | -7.0% | n/a | +60.0% | 0 |
| LLY | +18.1% | +62.1% | -7.9% | +66.7% | +33.3% | 0 |
| AVGO | +4.0% | +25.9% | -16.1% | +60.0% | +20.0% | 0 |
| JPM | +0.0% | +12.8% | -10.1% | n/a | n/a | 0 |
| V | +9.2% | +2.7% | +2.5% | +0.0% | +50.0% | 0 |
| XOM | +17.3% | +29.9% | +10.5% | +77.8% | +33.3% | 0 |
| UNH | +0.0% | +45.7% | +14.7% | n/a | n/a | 0 |
| MA | +0.0% | -8.7% | -4.0% | n/a | n/a | 0 |
| PG | +0.0% | -2.9% | +6.2% | n/a | n/a | 0 |
| JNJ | +0.0% | +49.0% | +4.1% | n/a | n/a | 0 |
| COST | +0.0% | -1.6% | +0.8% | n/a | n/a | 0 |
| HD | +0.0% | -15.2% | +4.9% | n/a | n/a | 0 |
| WMT | +0.0% | +20.1% | -7.0% | n/a | n/a | 0 |
| ORCL | +0.0% | -44.0% | +0.6% | n/a | n/a | 0 |
| MRK | +0.0% | +53.6% | -12.3% | n/a | n/a | 0 |
| ABBV | +7.0% | +22.0% | -7.9% | +0.0% | +0.0% | 0 |
| CVX | +11.4% | +17.5% | +6.6% | +90.9% | +33.3% | 0 |
| CRM | +0.0% | -32.4% | -20.3% | n/a | +50.0% | 0 |

## Caveats

- Fundamentals are today's data applied to historical weeks (Yahoo has no point-in-time fundamentals), so the fundamentals agent carries lookahead bias.
- No historical news is replayed; votes come from four agents.
- The universe is today's largest companies, which biases the study toward survivors.
- Weekly closes and opens come from Yahoo-adjusted daily bars; signals decide at a week's last session and execute at the next session's open.
- Decision quality does not establish profitability; this is a research measurement.
