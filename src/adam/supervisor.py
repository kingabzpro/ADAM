from pathlib import Path

from adam.config import load_config
from adam.dashboard import run_dashboard


def start() -> None:
    config = load_config(Path("config.yaml"))
    run_dashboard(config)
