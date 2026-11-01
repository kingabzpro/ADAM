# ADAM

**Local AI-powered stock research, with risk checks built in.**

[![Tests](https://github.com/kingabzpro/ADAM/actions/workflows/tests.yml/badge.svg)](https://github.com/kingabzpro/ADAM/actions/workflows/tests.yml) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/) [![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-555)](#get-started) [![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

ADAM combines market data, local forecasting, financial sentiment, and D3 decision models in a single research dashboard. Five specialist agents vote, and their consensus is the call you see. A deterministic risk engine still evaluates and records every analysis.

> **Work in progress:** News coverage, automated risk evidence, and historical evaluation are still being developed. ADAM is a research tool, not a brokerage or a promise of returns.

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --extra forecast --extra sentiment
uv run adam-setup
uv run adam
```

Setup detects Windows, macOS, or Linux, installs or updates [Ollama](https://ollama.com/) when needed, and downloads the local D3 model (about 3.3 GB). The official installer may request administrator privileges; on Linux it may configure a service and GPU dependencies. The first forecast or sentiment run also downloads its model.

Open http://127.0.0.1:8080, enter ticker symbols, and run analysis. The page streams each agent's progress as it finishes and shows the per-agent signals, consensus, and risk-gated result. Risk limits use conservative built-in defaults; the [risk input guide](docs/risk-input.md) covers supplying your own programmatically. Missing critical inputs produce **No Trade**. Analysis history is stored locally in `~/.adam/history.sqlite`.

Expand a ticker's evidence to see the vote breakdown, forecast quantiles, news sources, and risk inputs. D3 provides choice scores, not a causal explanation.

Concurrency settings are optional; see the [configuration guide](docs/configuration.md).
For platform requirements and troubleshooting, see the [installation guide](docs/install.md).

Online analyses save local data snapshots for [offline research](docs/offline.md) (`ADAM_OFFLINE=1`). Offline results always show **No Trade**.

For simulated single-ticker tracking and saved-Buy outcome evaluation, see [paper trading](docs/paper.md) and `uv run adam-outcomes TICKER`.

## Built with

- **Market data:** Yahoo Finance · **Forecasts:** Timestral · **Sentiment:** FinBERT
- **Decisions:** D3 via Ollama · **Interface:** FastAPI with a single static page and SSE streaming · **Runtime:** Python and uv

Model inference runs locally; market data, news searches, setup, and model downloads use the internet. News uses a [limited article source](docs/news.md) and falls back to headlines; results may be unrelated to the ticker. FinBERT weights are [CC BY-NC-SA 4.0](https://huggingface.co/ProsusAI/finbert); review their terms before use.

## Development

```bash
uv run python -m unittest discover -s tests
uv run adam-benchmark --requests 10 --concurrency 2
```

The benchmark requires the local D3 model. See the [testing guide](docs/testing.md) for the real-model check.

See [plan.md](plan.md) for the roadmap. ADAM code is licensed under [Apache 2.0](LICENSE); model licenses are separate.
