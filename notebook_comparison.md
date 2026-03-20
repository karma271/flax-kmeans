# Notebook Comparison Workflow

This document defines the Colab workflow for benchmarking all four KMeans implementations with strict comparability.

## Notebook layout

- `notebooks/01_sklearn_cpu.ipynb`: sklearn benchmark on CPU runtime.
- `notebooks/02_jax_kmeans_tpu.ipynb`: JAX KMeans benchmark on TPU runtime.
- `notebooks/03_flashkmeans_gpu.ipynb`: flash-kmeans wrapper benchmark on GPU runtime.
- `notebooks/04_jax_flash_kmeans_tpu.ipynb`: JAX flash-style KMeans benchmark on TPU runtime.
- `notebooks/05_comparison_summary.ipynb`: completeness check + comparative analysis summary.

## Shared benchmark matrix

All execution notebooks use:

- `configs/experiments/benchmark_matrix.yaml`

This ensures each implementation is run against the same dataset scales and seed list.

## Artifact destination

Each execution notebook writes artifacts to:

- `results/<exp_name>/manifests/<run_id>.json`
- `results/<exp_name>/metrics/<run_id>.json`

`<exp_name>` comes from the benchmark matrix file.

## Deterministic run IDs

Run IDs are generated as:

- `<exp_name>__<implementation>__<device>__<dataset_id>__<scale_id>__s<seed>`

This keeps run identity stable across Colab sessions and avoids accidental duplication.

## Colab run order

1. Run notebook 01 on CPU and commit/push artifacts.
2. Run notebook 02 on TPU and commit/push artifacts.
3. Run notebook 03 on GPU and commit/push artifacts.
4. Run notebook 04 on TPU and commit/push artifacts.
5. Run notebook 05 to validate completeness and produce comparative outputs.

## Comparison outputs

Notebook 05 writes standard outputs to `results/<exp_name>/`:

- `metrics/comparative_runs.csv`
- `metrics/summary_by_device_scale.csv`
- `metrics/summary_by_implementation.csv`
- `plots/tradeoff_speed_vs_quality.html`
- `plots/scaling_fit_time.html`
- `plots/device_total_time.html`
- `plots/tradeoff_report.md`
