"""Config I/O helpers for experiment benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.eval.contracts import ExperimentConfig
from src.eval.schema_validation import validate_experiment_config_payload


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one experiment config YAML file into the typed contract."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a YAML mapping/object.")
    casted_payload = cast_dict(payload)
    validate_experiment_config_payload(casted_payload)
    return ExperimentConfig(**casted_payload)


def cast_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Return dictionary with string keys for dataclass initialization."""
    return {str(k): v for k, v in value.items()}
