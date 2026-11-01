# Adaptive Decision Agent Manager (ADAM)
ADAM is a fully local, open-source AI hedge fund research platform that coordinates specialized decision agents for market analysis, price forecasting, news sentiment, and fundamentals. Powered by **Yahoo Finance, Mistral Timestral, FinBERT, and D3 via vLLM**, it generates consensus-based Buy/Hold/Sell signals with real-time analysis and built-in risk management—all without paid AI APIs.

Configuration currently defines `inference_concurrency` in `config.yaml`. The inference runtime will consume it when implemented.
