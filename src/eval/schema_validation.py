"""JSON schema validation helpers for benchmark contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments"


def _load_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Schema file must be a JSON object: {path}")
    return payload


_EXPERIMENT_CONFIG_VALIDATOR = Draft202012Validator(
    _load_schema(_SCHEMA_DIR / "experiment_config.schema.json")
)
_RUN_MANIFEST_VALIDATOR = Draft202012Validator(
    _load_schema(_SCHEMA_DIR / "run_manifest.schema.json")
)
_METRIC_RECORD_VALIDATOR = Draft202012Validator(
    _load_schema(_SCHEMA_DIR / "metrics_record.schema.json")
)


def _raise_validation_error(error: ValidationError, schema_name: str) -> None:
    path = ".".join(str(part) for part in error.path)
    location = path or "<root>"
    raise ValueError(
        f"{schema_name} schema validation failed at {location}: {error.message}"
    ) from error


def validate_experiment_config_payload(payload: dict[str, Any]) -> None:
    """Validate experiment config payload against JSON schema."""
    try:
        _EXPERIMENT_CONFIG_VALIDATOR.validate(payload)
    except ValidationError as error:
        _raise_validation_error(error, "ExperimentConfig")


def validate_run_manifest_payload(payload: dict[str, Any]) -> None:
    """Validate run manifest payload against JSON schema."""
    try:
        _RUN_MANIFEST_VALIDATOR.validate(payload)
    except ValidationError as error:
        _raise_validation_error(error, "RunManifest")


def validate_metric_record_payload(payload: dict[str, Any]) -> None:
    """Validate metric record payload against JSON schema."""
    try:
        _METRIC_RECORD_VALIDATOR.validate(payload)
    except ValidationError as error:
        _raise_validation_error(error, "MetricRecord")
