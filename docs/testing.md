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

For historical research, `adam.evaluation.compare_voting` compares majority and weighted votes using the same risk limits, transaction costs, and next-session execution. Supply one set of agent decisions and `RiskEvidence` for each timezone-aware price index timestamp, plus weights fixed before the evaluation period. The index timestamps must represent the decision times. Market and forecast decisions must both be available and reliable; otherwise the result is No Trade. The caller must ensure that every decision, input, price adjustment, and weight was available at that time: the comparison cannot detect look-ahead bias. It does not fit weights or recreate historical AI decisions.

The comparison invests at most the supplied `position_fraction` of available cash on a Buy, including fees; it does not rebalance an open position. A Sell closes the position. The standalone `backtest` defaults to investing all available cash unless `position_fractions` is supplied. Neither simulates a multi-asset portfolio, slippage, taxes, or corporate actions.
