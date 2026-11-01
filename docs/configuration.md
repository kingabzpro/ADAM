# Configuration

ADAM starts with built-in defaults when there is no `config.yaml` in the current directory. Create that file to override any of these settings:

```yaml
inference_concurrency: 1      # concurrent D3 requests
data_concurrency: 4           # concurrent Yahoo price and fundamentals calls per analysis run
ticker_concurrency: 4         # concurrent ticker analyses per dashboard run
yahoo_interval_seconds: 0.25  # minimum spacing between Yahoo requests (0 to 60)
```

All settings above are positive integers except `yahoo_interval_seconds`, which accepts any number between 0 and 60, including fractions.

For a configuration stored elsewhere, set `ADAM_CONFIG` to its path before starting `uv run adam`. An explicitly specified file must exist. Unsupported settings and invalid values stop startup with an error. Concurrency limits apply per dashboard analysis run, not globally across separate browser sessions or processes. Increasing them can increase memory use and pressure on external data providers.
