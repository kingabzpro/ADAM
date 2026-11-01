# Offline research

While online, run an analysis for each ticker you want to research later. ADAM saves validated daily prices, the SPY benchmark, current fundamentals, and usable news evidence to `~/.adam/snapshots`. Snapshot write failures appear in the analysis errors. Run `uv run adam-setup` online first so Ollama and `d3:2b` are already installed; a first online analysis also downloads the Timestral weights when the forecast extra is installed.

Set `ADAM_OFFLINE=1` before starting `uv run adam` to use those snapshots and cached Timestral weights. ADAM uses the saved sentiment evidence without loading FinBERT or fetching new articles; its decision is marked stale and abstains from voting. Saved-outcome evaluation still requires the internet. Missing snapshots or model weights are reported as errors; ADAM will not fetch replacements in this mode. The evidence includes snapshot times, and offline analysis always returns **No Trade** because saved data can be stale. Model inference still needs the local Ollama server.

Snapshots are stored as local files and can include market and company data. They are separate from `~/.adam/history.sqlite`. Online price-only loading does not create snapshots; run an analysis to save them. Offline behavior was checked on Windows with external socket connections blocked; other platforms still need live offline validation.
