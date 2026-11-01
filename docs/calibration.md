# Evaluating decision scores

Use independently labeled examples to check how an agent's choice scores relate to the labels. Create a UTF-8 JSON Lines file with one example per line. Each example contains the complete saved `Decision` object and a `label` of `BUY`, `HOLD`, or `SELL`:

```json
{"decision":{"ticker":"AAPL","agent":"market","signal":"BUY","probabilities":{"BUY":0.8,"HOLD":0.1,"SELL":0.1},"summary":"Example","status":"completed","reliable":true,"stale":false},"label":"BUY"}
```

Then run:

```bash
uv run adam-calibrate examples.jsonl --bins 10
```

The output reports accuracy, multiclass Brier score (sum of squared errors across all three choices), log loss, and top-choice expected calibration error with equal-width reliability bins. The last bin includes a score of 1. Log loss clips a zero score to `1e-15`. Stale, unreliable, and incomplete decisions are excluded; evaluate one agent category at a time. Decisions are visible in saved analysis history. The labels must be created independently using a clearly defined evaluation protocol, not copied from the model's own choices.

These diagnostics compare scores with the supplied labels. D3 scores express the model's choice among criteria; they are not probabilities of price direction, profit, or successful trades. A representative held-out dataset, independent labeling, and enough samples per bin are needed before interpreting calibration. The command does not verify label quality, temporal separation, or model training overlap, and it does not change model weights or trading decisions.
