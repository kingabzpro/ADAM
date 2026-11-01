from dataclasses import dataclass
from pathlib import Path
from math import isfinite

import yaml


@dataclass(frozen=True)
class Config:
    inference_concurrency: int = 1
    data_concurrency: int = 4
    ticker_concurrency: int = 4
    yahoo_interval_seconds: float = .25


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or set(data) - set(Config.__dataclass_fields__):
        raise ValueError("Configuration must contain only supported settings")
    for name, value in data.items():
        if name == 'yahoo_interval_seconds':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 60:
                raise ValueError('yahoo_interval_seconds must be between 0 and 60')
        elif type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return Config(**data)
