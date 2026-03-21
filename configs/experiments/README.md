# Experiment Configs and Schemas

These files define benchmark inputs and the required JSON artifact structure.

## Files

- `experiment_config.template.yaml`: single-run config template.
- `benchmark_matrix.yaml`: shared multi-run matrix used by notebooks.
- `experiment_config.schema.json`: schema for run configuration inputs.
- `run_manifest.schema.json`: schema for run metadata outputs.
- `metrics_record.schema.json`: schema for clustering metric outputs.

## Artifact paths

- Manifests: `results/manifests/<run_id>.json`
- Metrics: `results/metrics/<run_id>.json`

## Consistency rules

- `run_id` must match between manifest and metrics artifacts.
- `implementation` must be one of canonical implementation IDs.
- `dataset_id`, `n_clusters`, and `random_seed` are required for every run.
- Timing fields use milliseconds (`fit_time_ms`, `predict_time_ms`).

## Commands

Run one benchmark from config + `.npy` features:

```bash
uv run python -m src.eval.run_benchmark \
  --config configs/experiments/experiment_config.template.yaml \
  --x-npy /path/to/features.npy \
  --results-root results
```

Generate synthetic `.npy` inputs:

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

Pass implementation-specific kwargs as JSON:

```bash
uv run python -m src.eval.run_benchmark \
  --config configs/experiments/experiment_config.template.yaml \
  --x-npy /path/to/features.npy \
  --implementation-kwargs '{"data_chunk_size": 2048, "centroid_chunk_size": 128}'
```

Build comparative outputs from collected artifacts:

```bash
uv run python -m src.eval.run_comparative_analysis --results-root results --output-root results
```

Notebook runs can use `benchmark_matrix.yaml` and write to:

- `results/<exp_name>/manifests`
- `results/<exp_name>/metrics`
