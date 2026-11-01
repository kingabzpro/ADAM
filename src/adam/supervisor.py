import logging
import os
from pathlib import Path

from adam.config import load_config
from adam.dashboard import run_dashboard
from adam.setup import ensure_server, ollama_executable


def start() -> None:
    configured_path = os.environ.get('ADAM_CONFIG')
    default_path = Path('config.yaml')
    config = load_config(Path(configured_path) if configured_path is not None else default_path if default_path.exists() else None)
    executable = ollama_executable()
    if executable:
        try:
            ensure_server(executable)
        except (OSError, RuntimeError) as error:
            logging.warning('Ollama could not be started: %s', error)
    run_dashboard(config)
