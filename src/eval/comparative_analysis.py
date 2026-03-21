"""Comparative analysis helpers for reporting benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from src.plots.theme import use_default_template

MANIFEST_COLUMNS = {
    "run_id",
    "timestamp_utc",
    "implementation",
    "device",
    "dataset_id",
    "n_samples",
    "n_features",
    "n_clusters",
    "random_seed",
    "max_iter",
    "tolerance",
    "n_init",
    "converged",
    "iterations_used",
    "fit_time_ms",
    "predict_time_ms",
    "peak_memory_mb",
}

METRICS_COLUMNS = {
    "run_id",
    "implementation",
    "dataset_id",
    "inertia",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
}


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load all JSON object files from a directory."""
    if not path.exists():
        raise FileNotFoundError(f"Required directory does not exist: {path}")

    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {path}")

    records: list[dict[str, Any]] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected JSON object in {json_file}, found {type(payload).__name__}."
            )
        records.append(payload)
    return records


def load_artifact_frames(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read benchmark manifests and metrics artifacts into DataFrames."""
    manifests = pd.DataFrame(_load_json_records(results_root / "manifests"))
    metrics = pd.DataFrame(_load_json_records(results_root / "metrics"))

    missing_manifest = MANIFEST_COLUMNS - set(manifests.columns)
    missing_metrics = METRICS_COLUMNS - set(metrics.columns)
    if missing_manifest:
        raise ValueError(f"Manifest records missing required fields: {sorted(missing_manifest)}")
    if missing_metrics:
        raise ValueError(f"Metric records missing required fields: {sorted(missing_metrics)}")
    return manifests, metrics


def build_comparative_dataframe(manifests: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Join manifests and metrics and add convenience analysis columns."""
    manifests_run_id_duplicates = manifests["run_id"].duplicated()
    metrics_run_id_duplicates = metrics["run_id"].duplicated()
    if bool(manifests_run_id_duplicates.any()):
        duplicate_run_ids = manifests.loc[manifests_run_id_duplicates, "run_id"].tolist()
        raise ValueError(f"Manifest run_id values must be unique. Duplicates: {duplicate_run_ids}")
    if bool(metrics_run_id_duplicates.any()):
        duplicate_run_ids = metrics.loc[metrics_run_id_duplicates, "run_id"].tolist()
        raise ValueError(f"Metric run_id values must be unique. Duplicates: {duplicate_run_ids}")

    merged = manifests.merge(
        metrics,
        on="run_id",
        how="inner",
        suffixes=("_manifest", "_metric"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching run_id values across manifests and metrics.")

    incompatible_rows = (merged["implementation_manifest"] != merged["implementation_metric"]) | (
        merged["dataset_id_manifest"] != merged["dataset_id_metric"]
    )
    if bool(incompatible_rows.any()):
        mismatched_run_ids = merged.loc[incompatible_rows, "run_id"].tolist()
        raise ValueError(
            "Manifest/metric rows disagree on implementation or dataset_id "
            f"for run_ids: {mismatched_run_ids}"
        )

    comparative = merged.rename(
        columns={
            "implementation_manifest": "implementation",
            "dataset_id_manifest": "dataset_id",
        }
    ).drop(columns=["implementation_metric", "dataset_id_metric"])
    comparative["total_time_ms"] = comparative["fit_time_ms"] + comparative["predict_time_ms"]
    comparative["scale_id"] = (
        comparative["n_samples"].astype(str)
        + "x"
        + comparative["n_features"].astype(str)
        + "_k"
        + comparative["n_clusters"].astype(str)
    )
    comparative["timestamp_utc"] = pd.to_datetime(comparative["timestamp_utc"], utc=True)
    return comparative.sort_values(
        ["dataset_id", "device", "implementation", "run_id"]
    ).reset_index(drop=True)


def build_summary_tables(comparative: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build grouped summary tables for cross-implementation comparisons."""
    by_device_scale = (
        comparative.groupby(["dataset_id", "device", "scale_id", "implementation"], dropna=False)
        .agg(
            run_count=("run_id", "count"),
            fit_time_ms_mean=("fit_time_ms", "mean"),
            fit_time_ms_median=("fit_time_ms", "median"),
            fit_time_ms_std=("fit_time_ms", "std"),
            predict_time_ms_mean=("predict_time_ms", "mean"),
            predict_time_ms_median=("predict_time_ms", "median"),
            predict_time_ms_std=("predict_time_ms", "std"),
            total_time_ms_mean=("total_time_ms", "mean"),
            total_time_ms_median=("total_time_ms", "median"),
            total_time_ms_std=("total_time_ms", "std"),
            iterations_used_mean=("iterations_used", "mean"),
            iterations_used_median=("iterations_used", "median"),
            inertia_mean=("inertia", "mean"),
            inertia_median=("inertia", "median"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_median=("silhouette", "median"),
            calinski_harabasz_mean=("calinski_harabasz", "mean"),
            calinski_harabasz_median=("calinski_harabasz", "median"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            davies_bouldin_median=("davies_bouldin", "median"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
            peak_memory_mb_median=("peak_memory_mb", "median"),
        )
        .reset_index()
        .sort_values(["dataset_id", "device", "scale_id", "implementation"])
    )

    by_implementation = (
        comparative.groupby(["implementation", "device"], dropna=False)
        .agg(
            run_count=("run_id", "count"),
            datasets_seen=("dataset_id", "nunique"),
            scales_seen=("scale_id", "nunique"),
            fit_time_ms_median=("fit_time_ms", "median"),
            predict_time_ms_median=("predict_time_ms", "median"),
            total_time_ms_median=("total_time_ms", "median"),
            inertia_median=("inertia", "median"),
            silhouette_median=("silhouette", "median"),
            calinski_harabasz_median=("calinski_harabasz", "median"),
            davies_bouldin_median=("davies_bouldin", "median"),
            peak_memory_mb_median=("peak_memory_mb", "median"),
        )
        .reset_index()
        .sort_values(
            ["device", "total_time_ms_median", "silhouette_median"], ascending=[True, True, False]
        )
    )
    return {
        "comparative_runs": comparative,
        "summary_by_device_scale": by_device_scale,
        "summary_by_implementation": by_implementation,
    }


def build_comparative_figures(comparative: pd.DataFrame) -> dict[str, go.Figure]:
    """Build standard comparative analysis figures for performance and quality."""
    use_default_template()

    tradeoff_fig = px.scatter(
        comparative,
        x="fit_time_ms",
        y="silhouette",
        color="implementation",
        symbol="device",
        hover_data=["run_id", "dataset_id", "scale_id", "inertia", "davies_bouldin"],
        title="Speed vs Quality Tradeoff (Fit Time vs Silhouette)",
    )
    tradeoff_fig.update_xaxes(type="log", title="Fit Time (ms, log scale)")
    tradeoff_fig.update_yaxes(title="Silhouette")

    scaling_summary = (
        comparative.groupby(["dataset_id", "device", "implementation", "n_samples"], dropna=False)
        .agg(fit_time_ms_median=("fit_time_ms", "median"))
        .reset_index()
        .sort_values(["dataset_id", "device", "implementation", "n_samples"])
    )
    scaling_fig = px.line(
        scaling_summary,
        x="n_samples",
        y="fit_time_ms_median",
        color="implementation",
        line_dash="device",
        facet_col="dataset_id",
        markers=True,
        title="Scaling Curve (Median Fit Time vs Samples)",
    )
    scaling_fig.update_yaxes(type="log", title="Median Fit Time (ms, log scale)")
    scaling_fig.update_xaxes(title="n_samples")

    device_summary = (
        comparative.groupby(["device", "implementation"], dropna=False)
        .agg(total_time_ms_median=("total_time_ms", "median"))
        .reset_index()
        .sort_values(["device", "total_time_ms_median"])
    )
    device_fig = px.bar(
        device_summary,
        x="device",
        y="total_time_ms_median",
        color="implementation",
        barmode="group",
        title="Device Comparison (Median Total Time)",
    )
    device_fig.update_yaxes(type="log", title="Median Total Time (ms, log scale)")
    device_fig.update_xaxes(title="Device")

    return {
        "tradeoff_speed_vs_quality": tradeoff_fig,
        "scaling_fit_time": scaling_fig,
        "device_total_time": device_fig,
    }


def build_tradeoff_report(summary_by_device_scale: pd.DataFrame) -> str:
    """Generate a concise markdown report of fast-vs-quality winners."""
    if summary_by_device_scale.empty:
        return "# Tradeoff Report\n\nNo comparative rows were available.\n"

    lines = [
        "# Tradeoff Report",
        "",
        "This report highlights median time and quality winners per dataset/device/scale.",
        "",
    ]
    group_cols = ["dataset_id", "device", "scale_id"]
    for (dataset_id, device, scale_id), group in summary_by_device_scale.groupby(
        group_cols, dropna=False
    ):
        fastest = group.loc[group["total_time_ms_median"].idxmin()]
        valid_quality = group[group["silhouette_median"].notna()]
        quality = (
            valid_quality.loc[valid_quality["silhouette_median"].idxmax()]
            if not valid_quality.empty
            else None
        )
        lines.append(f"## {dataset_id} | {device} | {scale_id}")
        lines.append(
            f"- Fastest median total time: `{fastest['implementation']}` "
            f"({fastest['total_time_ms_median']:.3f} ms)"
        )
        if quality is None:
            lines.append("- Best median silhouette: unavailable (all values missing).")
        else:
            lines.append(
                f"- Best median silhouette: `{quality['implementation']}` "
                f"({quality['silhouette_median']:.4f})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_analysis_outputs(
    summary_tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
    *,
    output_root: Path,
) -> dict[str, Path]:
    """Persist summary tables and figures to disk."""
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for name, frame in summary_tables.items():
        table_path = metrics_dir / f"{name}.csv"
        frame.to_csv(table_path, index=False)
        output_paths[name] = table_path

    for name, figure in figures.items():
        plot_path = plots_dir / f"{name}.html"
        figure.write_html(plot_path)
        output_paths[name] = plot_path

    report_path = plots_dir / "tradeoff_report.md"
    report_path.write_text(
        build_tradeoff_report(summary_tables["summary_by_device_scale"]),
        encoding="utf-8",
    )
    output_paths["tradeoff_report"] = report_path
    return output_paths
