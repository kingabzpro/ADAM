# Installation and hardware

ADAM runs natively on Windows, macOS, and Linux. Install [Git](https://git-scm.com/install/) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands in PowerShell or a terminal:

```bash
git clone https://github.com/kingabzpro/ADAM.git
cd ADAM
uv sync --extra forecast --extra sentiment
uv run adam-setup
uv run adam
```

`adam-setup` detects the operating system, installs or updates Ollama if needed, starts its local server, and downloads `d3:2b`. On Windows it uses native Ollama; WSL is not required. On macOS and Linux it uses Ollama's official installer, which may request administrator credentials and modify system services. The web page checks whether the server and model are available when opened; reload the page to refresh the status. The check does not run a model inference.

The first forecast and news analyses download Timestral and FinBERT weights respectively. Omit either `--extra` from `uv sync` if you do not need that component. Missing forecast data prevents a trade recommendation. See [offline research](offline.md) for preparing local snapshots and model weights before disconnecting.

## Capacity and platform notes

- `d3:2b` downloads roughly 3.2 GB; allow additional free space for Ollama, Python packages, model weights, and temporary installation files. The [Ollama Windows guide](https://docs.ollama.com/windows) lists at least 4 GB for Ollama itself.
- Ollama reports testing `d3:2b` on an 8 GB Apple Silicon system with approximately 3.62 GB of loaded model memory at a 4,096-token context. ADAM's forecast, sentiment, dashboard, and other software require additional memory. Actual requirements vary by platform and workload; see the [model page](https://ollama.com/library/d3:2b).
- CPU-only inference is possible but can be slow. Ollama supports Apple Silicon acceleration and supported NVIDIA and AMD GPUs; consult its [hardware guide](https://docs.ollama.com/gpu) for current driver and device requirements. Ollama lists macOS 14 or newer and Windows 10 22H2 or newer in its [macOS](https://docs.ollama.com/macos) and [Windows](https://docs.ollama.com/windows) guides.
- The Windows installation and local D3 path have been tested on this project. Linux and macOS setup paths have automated mocked coverage, but full installation and GPU behavior still need testing on those platforms.

## Troubleshooting

- **Ollama unavailable or model missing:** Run `uv run adam-setup` again and reload the ADAM page to recheck readiness. If an outdated server is still running after an update, restart Ollama.
- **Forecast or news component missing:** Run `uv sync --extra forecast --extra sentiment` from the project directory, then restart ADAM.
- **Offline snapshot missing:** Run an online analysis for that ticker first. Offline data is research-only and always produces No Trade.
- **Market or news source unavailable:** Retry later; external sources can rate-limit or fail. Check the individual agent errors in the dashboard.

See the [configuration guide](configuration.md) for concurrency settings and the [testing guide](testing.md) for a real local-model check.
