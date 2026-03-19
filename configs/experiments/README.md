# Experiment Contracts

All implementations must emit artifacts using these contracts to ensure comparisons are valid.

## Files

- `experiment_config.template.yaml`: template used to launch a run.
- `experiment_config.schema.json`: schema for experiment inputs.
- `run_manifest.schema.json`: schema for execution metadata written per run.
- `metrics_record.schema.json`: schema for normalized clustering quality metrics.

## Artifact locations

- Manifests: `results/manifests/<run_id>.json`
- Metrics: `results/metrics/<run_id>.json`

## Required comparability guarantees

- Shared `run_id` between manifest and metrics outputs.
- `implementation` uses one of the canonical IDs.
- `dataset_id`, `n_clusters`, and `random_seed` must be captured for every run.
- Time units are milliseconds (`fit_time_ms`, `predict_time_ms`).
