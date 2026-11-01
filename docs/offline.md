# Offline research

While online, run an analysis for each ticker you want to research later. ADAM saves validated daily prices, the SPY benchmark, and current fundamentals to `~/.adam/snapshots`. Snapshot write failures appear in the analysis errors. Run `uv run adam-setup` online first so Ollama and `d3:2b` are already installed. If using forecasts, run an online analysis with Timestral enabled first to download its weights.

Select **Offline** in the dashboard to use those snapshots and cached Timestral weights. Disable the news checkbox: news and saved-outcome evaluation require the internet. Missing snapshots or model weights are reported as errors; ADAM will not fetch replacements in this mode. The evidence includes snapshot times, and offline analysis always returns **No Trade** because saved data can be stale. Model inference still needs the local Ollama server.

Snapshots are stored as local files and can include market and company data. They are separate from `~/.adam/history.sqlite`. Online price-only loading does not create snapshots; run an analysis to save them. Offline behavior was checked on Windows with external socket connections blocked; other platforms still need live offline validation.
