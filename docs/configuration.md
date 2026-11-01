# Configuration

ADAM starts with built-in defaults when there is no `config.yaml` in the current directory. Create that file to override any of these positive integer settings:

```yaml
inference_concurrency: 1  # concurrent D3 requests
data_concurrency: 4       # concurrent Yahoo price and fundamentals calls per analysis run
ticker_concurrency: 4     # concurrent ticker analyses per dashboard run
```

For a configuration stored elsewhere, set `ADAM_CONFIG` to its path before starting `uv run adam`. An explicitly specified file must exist. Unsupported settings and invalid values stop startup with an error. Concurrency limits apply per dashboard analysis run, not globally across separate browser sessions or processes. Increasing them can increase memory use and pressure on external data providers.
