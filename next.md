# Next High-Impact Steps

This project already has a solid core: multiple KMeans backends, validated run artifacts, and comparative analysis outputs. The next highest-leverage work is to improve repeatability at scale, benchmark realism, and measurement trust.

## 1) Add a matrix/sweep CLI (notebook-independent execution)

### Why this is interesting now

- You already have a strong matrix definition in `configs/experiments/benchmark_matrix.yaml`.
- Matrix execution exists in `src/eval/notebook_harness.py`, but it is notebook-oriented and less convenient for CI/headless reruns.
- A dedicated CLI unlocks reproducible, automation-friendly runs across machines and schedules.

### What good looks like

- One command runs an entire matrix for one or many implementations.
- Runs can resume safely (skip existing artifacts unless overwrite is requested).
- Teams can launch full benchmark suites without editing notebooks or relying on notebook state.

### High-level breakdown

1. Create a new CLI entrypoint (for example `src/eval/run_benchmark_matrix.py`) that loads the same matrix schema used by the harness.
2. Reuse existing orchestration logic from `run_matrix_for_implementation` to avoid duplicate behavior.
3. Add CLI flags for implementation selection, results root, and overwrite policy.
4. Emit a concise run summary (written/skipped/failed counts) to make long sweeps easier to monitor.
5. Add tests for CLI argument handling, skip/overwrite behavior, and artifact completeness.

### Risk and effort

- **Effort:** Medium.
- **Primary risk:** Drift between notebook and CLI logic if code paths diverge.
- **Mitigation:** Centralize matrix-run logic in shared functions and keep notebooks as thin wrappers.

---

## 2) Implement real dataset ingestion and preprocessing pipeline

### Why this is interesting now

- `configs/datasets/README.md` defines meaningful real-dataset plans (`pen_digits`, `wholesale_customers`), but the implementation is mostly conceptual.
- Current benchmarking leans heavily on synthetic data; real-world distributions can expose different convergence and quality behavior.
- A reproducible ingestion pipeline makes comparisons credible and easier to reproduce later.

### What good looks like

- Dataset acquisition + preprocessing is scriptable and deterministic.
- Processed artifacts land in a consistent location/format (e.g., `.npy` features and optional labels).
- Benchmark configs can reference real datasets as first-class inputs, not manual one-off prep.

### High-level breakdown

1. Add dataset-specific ingestion scripts or a unified dataset CLI under `src/data/`.
2. Implement deterministic preprocessing per dataset plan (feature selection, transforms, scaling, seed control).
3. Write processed outputs and metadata (shape, preprocessing choices, optional source hash/version) to stable paths.
4. Add dataset config wiring so experiment configs can point to prepared real dataset artifacts.
5. Add tests for data shape/type invariants and deterministic output checks.

### Risk and effort

- **Effort:** Medium to high.
- **Primary risk:** External data source instability (URL changes, format drift).
- **Mitigation:** Versioned dataset metadata, clear caching conventions, and robust validation at ingest time.

---

## 3) Improve benchmark trustworthiness and provenance

### Why this is interesting now

- Comparative outputs are already useful, but small measurement/provenance gaps can reduce confidence in close-call results.
- Stronger metadata and timing discipline make longitudinal comparisons more defensible.
- This work compounds over time because every future run becomes easier to audit.

### What good looks like

- Each run records enough context to explain changes (code version, dependency versions, effective kwargs, dataset identity).
- Timing metrics are less noisy (warmup/repeats and clear metric naming).
- Analysis consumes cleaner, explicitly scoped artifact sets.

### High-level breakdown

1. Extend run metadata to include stronger provenance (for example commit hash, matrix identity, and resolved implementation kwargs).
2. Add configurable timing discipline: warmup passes plus repeated measured runs with median/dispersion fields.
3. Clarify or split timing fields where needed (e.g., fit time vs shared NumPy assignment time) to avoid ambiguous interpretation.
4. Tighten comparative ingestion with explicit filtering/scoping and stronger validation of loaded artifacts.
5. Surface provenance fields in summary tables/reports so trend analysis is explainable.

### Risk and effort

- **Effort:** Medium.
- **Primary risk:** Schema evolution complexity for existing artifacts.
- **Mitigation:** Backward-compatible schema additions and migration-safe analysis logic.

---

## Suggested execution order

1. Matrix/sweep CLI (fastest path to operational leverage).
2. Trust/provenance improvements (improves reliability of all subsequent results).
3. Real dataset pipeline (higher lift, highest realism payoff).
