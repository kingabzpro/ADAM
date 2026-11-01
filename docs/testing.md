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
