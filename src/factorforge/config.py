from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "FactorForge", "language": "zh"},
    "data": {
        "source": "simulated",
        "start_date": "2023-01-02",
        "periods": 260,
        "symbols": 30,
        "seed": 42,
    },
    "research": {
        "forward_periods": 1,
        "quantiles": 5,
        "min_cross_section": 8,
    },
    "backtest": {"top_k": 5, "fee_bps": 10.0, "annualization": 252},
    "llm": {
        "mode": "mock",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    return _deep_merge(config, overrides)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
