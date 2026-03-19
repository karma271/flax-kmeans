# flax-kmeans

KMeans implementation and benchmarking project comparing:
- JAX KMeans (from scratch)
- JAX Flash-style KMeans
- sklearn KMeans baseline
- `flash-kmeans` package baseline

The goal is correctness and reproducibility first, then performance comparison across CPU/GPU/TPU environments.

## Initial Project Structure

```text
flax-kmeans/
  configs/
    datasets/
    experiments/
  notebooks/
  plans/
  results/
    manifests/
    metrics/
    plots/
  src/
    algorithms/
    data/
    eval/
    plots/
```

### Folder Responsibilities

- `src/algorithms/`: one module per implementation path (`jax_kmeans`, `jax_flash_kmeans`, `sklearn_kmeans`, `flashkmeans_wrapper`).
- `src/data/`: synthetic data generation and real dataset loading/preprocessing.
- `src/eval/`: clustering quality metrics and benchmark runners.
- `src/plots/`: Plotly theming and plotting utilities.
- `configs/datasets/`: dataset definitions and preprocessing options.
- `configs/experiments/`: run-time experiment configurations.
- `results/manifests/`: run metadata (device, seed, dataset, dimensions, k, timing).
- `results/metrics/`: normalized benchmark outputs for all implementations.
- `results/plots/`: generated plots and comparison figures.
- `notebooks/`: Colab-oriented analysis and execution notebooks.
- `plans/`: planning and design docs.

## Execution Workflow

1. Develop and test locally in Cursor (CPU-first for correctness).
2. Push commits to GitHub.
3. Pull/sync in Colab for GPU/TPU experiments.
4. Save all outputs into normalized artifacts under `results/`.

## Local Setup (uv)

```bash
# Local macOS setup
uv sync --extra dev --extra jax

# Linux/Colab setup (includes flash-kmeans)
uv sync --extra dev --extra jax --extra flash
```

Run core checks:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

## Contracts and Configs

- Experiment input template: `configs/experiments/experiment_config.template.yaml`
- Experiment input schema: `configs/experiments/experiment_config.schema.json`
- Run manifest schema: `configs/experiments/run_manifest.schema.json`
- Metrics schema: `configs/experiments/metrics_record.schema.json`
- Dataset plan and acquisition/preprocessing notes: `configs/datasets/README.md`

## Plot Styling Defaults

- Global style config: `configs/plot_style.yaml`
- Plotly template helper: `src/plots/theme.py`
- Recommended usage at notebook/script startup:
  - `from src.plots.theme import use_default_template`
  - `use_default_template()`

## Starter Checklist

- [ ] Create local environment with `uv sync --extra dev --extra jax`
- [ ] In Linux/Colab, include flash with `uv sync --extra dev --extra jax --extra flash`
- [ ] Implement and validate `jax_kmeans` + `sklearn_kmeans` on synthetic small cases
- [ ] Write run manifests to `results/manifests/` for every execution
- [ ] Write normalized metric records to `results/metrics/`
- [ ] Run GPU/TPU variants from Colab with GitHub sync
- [ ] Generate comparison plots in `results/plots/`

## Documentation References

- [JAX docs](https://docs.jax.dev/en/latest/)
- [flash-kmeans repository](https://github.com/svg-project/flash-kmeans)
- [flash-kmeans paper](https://arxiv.org/pdf/2603.09229)
