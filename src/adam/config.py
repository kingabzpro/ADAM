from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    inference_concurrency: int


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or set(data) != {"inference_concurrency"}:
        raise ValueError("Expected inference_concurrency in configuration")
    value = data["inference_concurrency"]
    if type(value) is not int or value < 1:
        raise ValueError("inference_concurrency must be a positive integer")
    return Config(inference_concurrency=value)
