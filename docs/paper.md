# Local paper trading

ADAM can maintain a single-ticker simulated cash account for research. It does not connect to a broker or place orders. Paper balances and fills are stored alongside analysis history in `~/.adam/history.sqlite`.

In the dashboard, open **Paper trading — simulation only**, choose a ticker, starting cash, and a per-side fee, then create an account. Load **Saved analysis history** to find the analysis ID. Enter that ID and a hypothetical price, then select **Record simulated trade**. The selected paper ticker must match the saved analysis; offline mode does not record simulated trades. Use **Load paper account** to view balances and recent simulated fills.

The command line is also available. Create an account once per ticker, with starting cash and an optional per-side transaction cost in basis points:

```bash
uv run adam-paper init AAPL --cash 10000 --cost-bps 10
```

Run a fresh online analysis with risk limits and evidence configured. In **Saved analysis history**, load and select the record to find its `id`. To simulate that record's approved Buy or Sell at a price you provide:

```bash
uv run adam-paper trade 123 --price 100.00
uv run adam-paper status AAPL --price 100.00
uv run adam-paper history AAPL
```

Replace the example ID and price with the saved record and a hypothetical fill price. A record can be used only once. ADAM checks that the saved result approved the action, the market and forecast decisions and evidence are present, the market comparison is available, and the analysis and market timestamp remain within the configured age limit. It recalculates consensus and the risk gate at the time of the simulation, accounting for the simulated account's resulting exposure and configured fee. Missing or failed checks reject the fill. Offline results are No Trade and cannot be used.

The account is long/cash only, holds at most one position, and uses the saved `position_fraction` to size a Buy from available cash, including the entry fee. A Sell closes the entire position. A mark price is required to calculate equity while holding shares. Trade history includes the rechecked risk inputs and limits. Duplicate execution is prevented transactionally.

Prices are manually entered and are not verified against a live market. Portfolio correlation, loss assumptions, and other supplied risk inputs are not independently verified; the simulator does not model slippage, taxes, dividends, corporate actions, partial fills, or multi-asset portfolios. Unrealized P&L excludes a hypothetical exit fee. Results are simulated and do not establish future trading performance.
