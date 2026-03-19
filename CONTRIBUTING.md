# Contributing Guidelines

## Environment

- Use `uv` for dependency management and virtual environments.
- Use Python 3.11+.
- Keep local development CPU-first; use Colab for GPU/TPU benchmarks.

## Coding Standards

- Prefer simple functional code over unnecessary abstractions.
- Use modern built-in type annotations (`list[str]`, `dict[str, int]`, etc.).
- Add docstrings for public functions and methods.
- Keep comments high-value and concise.
- Run linting and type checks before committing.

## Commands

- `uv sync --extra dev --extra jax` (local macOS)
- `uv sync --extra dev --extra jax --extra flash` (Linux/Colab)
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest`

## Plotting

- Use the project Plotly theme in `src/plots/theme.py`.
- Keep palette and font aligned with `configs/plot_style.yaml`.
