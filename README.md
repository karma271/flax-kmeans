# flax-kmeans

Small benchmarking project for KMeans implementations with reproducible artifacts.

Implemented backends:
- `jax_kmeans`
- `jax_flash_kmeans`
- `sklearn_kmeans`
- `flashkmeans_wrapper` (Linux/Colab)

## Quick Start

```bash
# macOS
uv sync --extra dev --extra jax

# Linux / Colab (includes flash-kmeans wrapper deps)
uv sync --extra dev --extra jax --extra flash
```

## Core Commands

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

Run one benchmark from a config + `.npy` feature matrix:

```bash
uv run python -m src.eval.run_benchmark \
  --config configs/experiments/experiment_config.template.yaml \
  --x-npy /path/to/features.npy \
  --results-root results
```

Generate comparative tables and plots from collected artifacts:

```bash
uv run python -m src.eval.run_comparative_analysis \
  --results-root results \
  --output-root results
```

## Project Reference

<details>
<summary><strong>Show layout and artifact outputs</strong></summary>

### Repository Layout

| Path | Purpose |
| --- | --- |
| `src/algorithms` | Implementation wrappers and fit entry points. |
| `src/eval` | Benchmark runner, schema validation, and comparative analysis. |
| `src/data` | Synthetic dataset generation utilities. |
| `src/plots` | Shared Plotly theme helper. |
| `configs/experiments` | Config templates and JSON schemas. |
| `configs/datasets` | Dataset notes and presets. |
| `notebooks` | Execution notebooks for per-device runs. |
| `results` | Generated artifacts (`manifests`, `metrics`, `plots`). |

### Artifacts

| Artifact | Description |
| --- | --- |
| `results/manifests/<run_id>.json` | Per-run execution metadata. |
| `results/metrics/<run_id>.json` | Per-run clustering quality metrics. |
| `results/metrics/comparative_runs.csv` | Joined run-level comparative table. |
| `results/metrics/summary_by_device_scale.csv` | Aggregated summary by dataset/device/scale/implementation. |
| `results/metrics/summary_by_implementation.csv` | Aggregated summary by implementation/device. |
| `results/plots/tradeoff_speed_vs_quality.html` | Speed vs quality scatter view. |
| `results/plots/scaling_fit_time.html` | Fit-time scaling curves. |
| `results/plots/device_total_time.html` | Device-level total-time comparison. |
| `results/plots/tradeoff_report.md` | Markdown summary of fastest/quality winners. |

</details>

## References

- [JAX docs](https://docs.jax.dev/en/latest/)
- [flash-kmeans repository](https://github.com/svg-project/flash-kmeans)
