#!/usr/bin/env python3
"""Pixel and soft-crest precision/recall curves for the selected model."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_all_configs, resolve_path
from .evaluate import (
    THRESHOLDS,
    cache_directory,
    load_targets,
    threshold_curve,
    threshold_histograms,
    write_csv,
)
from .infer import read_samples
from metrics.crest import soft_crest_counts


def source_crest_curve(cache, record) -> np.ndarray:
    probability = np.load(cache / f"{record['source_id']}.npy", mmap_mode="r")
    values = np.zeros((len(THRESHOLDS), 3), dtype=np.float64)
    for index, threshold in enumerate(THRESHOLDS):
        counts = soft_crest_counts(
            (probability > threshold) & record["valid"],
            target_features=record["target_crests"],
        )
        values[index] = (
            int(counts["crest_predicted"]),
            int(counts["crest_labeled"]),
            float(counts["crest_quality_sum"]),
        )
    return values


def source_crest_curve_task(arguments) -> np.ndarray:
    """Pickle-safe process-pool entry point for one complete image."""
    cache, record = arguments
    return source_crest_curve(cache, record)


def run_operating_points(workers: int = 8) -> None:
    manifest, entries = load_all_configs()
    entry, config = next(pair for pair in entries if pair[0].get("operating_points"))
    output_root = resolve_path(manifest["output_root"])
    cache = cache_directory(output_root / "cache", "validation", config["id"])
    dataset = resolve_path(config["dataset_root"])
    samples = read_samples(dataset, "val")
    targets = load_targets(dataset, samples)
    pixel_rows = threshold_curve(*threshold_histograms(cache, targets))

    # Crest extraction/matching contains substantial Python work, so threads
    # leave most allocated CPUs idle. Each image is independent and coarse
    # enough to amortize process startup and record serialization.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        source_values = list(
            executor.map(
                source_crest_curve_task,
                ((cache, record) for record in targets),
                chunksize=1,
            )
        )
    totals = np.sum(source_values, axis=0)
    crest_rows = []
    for index, threshold in enumerate(THRESHOLDS):
        predicted, labeled, quality = totals[index]
        crest_rows.append({
            "threshold": float(threshold),
            "crest_predicted": int(predicted),
            "crest_labeled": int(labeled),
            "crest_quality_sum": float(quality),
            "crest_precision": quality / predicted if predicted else float("nan"),
            "crest_recall": quality / labeled if labeled else float("nan"),
            "crest_f1": 2 * quality / (predicted + labeled)
            if predicted + labeled
            else float("nan"),
        })

    output = output_root / "metrics" / "operating_points"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pixel_precision_recall.csv", pixel_rows)
    write_csv(output / "crest_precision_recall.csv", crest_rows)
    thresholds = json.loads((output_root / "metrics" / "thresholds.json").read_text())
    selected = float(thresholds[config["id"]])
    selected_index = int(round(selected * 100)) - 1

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(
        [row["recall"] for row in pixel_rows],
        [row["precision"] for row in pixel_rows],
        color="#1769aa",
    )
    axes[0].scatter(
        pixel_rows[selected_index]["recall"],
        pixel_rows[selected_index]["precision"],
        color="#d1495b",
        label=f"selected threshold = {selected:.2f}",
        zorder=3,
    )
    axes[0].set(title="Pixel operating points", xlabel="Recall", ylabel="Precision")
    axes[1].plot(
        [row["crest_recall"] for row in crest_rows],
        [row["crest_precision"] for row in crest_rows],
        color="#2a9d8f",
    )
    axes[1].scatter(
        crest_rows[selected_index]["crest_recall"],
        crest_rows[selected_index]["crest_precision"],
        color="#d1495b",
        label=f"selected threshold = {selected:.2f}",
        zorder=3,
    )
    axes[1].set(
        title="Quality-weighted crest operating points",
        xlabel="Crest recall",
        ylabel="Crest precision",
    )
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left")
    figure.savefig(output / "operating_points.pdf")
    figure.savefig(output / "operating_points.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_operating_points(args.workers)


if __name__ == "__main__":
    main()
