# Dataset Plan (v1)

This project starts with one synthetic benchmark generator and two real datasets.

## Synthetic: `synthetic_blobs_v1.yaml`

Purpose:
- Controlled benchmarking for correctness, convergence behavior, and runtime scaling.

Sweep dimensions:
- `n_samples`: [1000, 10000, 100000]
- `n_features`: [8, 32, 128]
- `n_clusters`: [4, 8, 16]
- `cluster_std`: [0.3, 0.8, 1.5]
- `imbalance_ratio`: [1.0, 2.0, 5.0]
- `noise_fraction`: [0.0, 0.01, 0.05]

Generation notes:
- Use deterministic random seeds.
- Scale features before clustering.
- Keep a fixed sweep order for reproducible benchmark tables.

## Real: `pen_digits.yaml`

Dataset:
- UCI Pen-Based Recognition of Handwritten Digits.

Acquisition:
- Download from UCI repository and store raw files under `data/raw/pen_digits/`.

Preprocessing:
- Keep numeric coordinate features only.
- Standardize features.
- Preserve labels for optional external validation (not core metric set).

Why included:
- Medium-size, labeled, and well-suited for centroid-based clustering stress tests.

## Real: `wholesale_customers.yaml`

Dataset:
- UCI Wholesale Customers.

Acquisition:
- Download from UCI repository and store raw files under `data/raw/wholesale_customers/`.

Preprocessing:
- Use seven annual spending numeric features.
- Apply `log1p` transformation due to heavy skew.
- Standardize features after log transform.
- Keep `Channel` and `Region` for optional external diagnostics.

Why included:
- Small, business-like tabular distribution; useful for fast correctness and stability checks.
