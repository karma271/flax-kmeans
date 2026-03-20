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

## Minimal CLI usage

Run one config with a precomputed feature matrix (`.npy`):

```bash
uv run python -m src.eval.run_benchmark \
  --config configs/experiments/experiment_config.template.yaml \
  --x-npy /path/to/features.npy \
  --results-root results
```

Generate a quick synthetic `.npy` locally:

```bash
uv run python -m src.data.generate_synthetic_npy \
  --x-out results/tmp/synth_x.npy \
  --y-out results/tmp/synth_y.npy \
  --n-samples 2000 \
  --n-features 32 \
  --n-clusters 8 \
  --cluster-std 0.8 \
  --random-seed 42
```

Optional implementation-specific kwargs can be passed as JSON:

```bash
uv run python -m src.eval.run_benchmark \
  --config configs/experiments/experiment_config.template.yaml \
  --x-npy /path/to/features.npy \
  --implementation-kwargs '{"data_chunk_size": 2048, "centroid_chunk_size": 128}'
```

After multiple runs have produced JSON artifacts, generate comparative Phase E outputs:

```bash
uv run python -m src.eval.run_comparative_analysis --results-root results --output-root results
```
