import logging
from pathlib import Path

from adam.config import load_config
from adam.dashboard import run_dashboard
from adam.setup import ensure_server, ollama_executable


def start() -> None:
    config = load_config(Path('config.yaml'))
    executable = ollama_executable()
    if executable:
        try:
            ensure_server(executable)
        except (OSError, RuntimeError) as error:
            logging.warning('Ollama could not be started: %s', error)
    run_dashboard(config)
