#!/usr/bin/env python3
"""Plot complete-image pixel errors and one-to-one crest matches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import load_all_configs, resolve_path
from .evaluate import cache_directory, load_targets, write_csv
from .infer import read_samples
from metrics.crest import (
    extract_event_features,
    maximum_quality_assignment,
    ratios,
    score_feature_matrix,
    soft_crest_counts,
)
from metrics.pixel import metrics as pixel_metrics


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_quantile_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Select deterministic crest-F1 quantiles, including the range endpoints."""
    if count < 1:
        raise ValueError("count must be positive")
    if not rows:
        raise ValueError("No per-image metric rows available")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["crest_f1"])
            if np.isfinite(float(row["crest_f1"]))
            else -1.0,
            row["source_id"],
        ),
    )
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered)))
    return [ordered[int(round(index))] for index in indices]


def pixel_error_image(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    image = np.zeros((*target.shape, 3), dtype=np.float32)
    image[~valid] = (0.18, 0.18, 0.18)
    image[prediction & target] = (0.20, 0.90, 0.20)
    image[prediction & ~target & valid] = (1.00, 0.20, 0.20)
    image[~prediction & target & valid] = (0.20, 0.40, 1.00)
    return image


def crest_image(predicted, labeled, shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros((*shape, 3), dtype=np.float32)
    for feature in labeled:
        rows, columns = feature.coordinates.T
        image[rows, columns, 1:] = 1.0  # cyan labels
    for feature in predicted:
        rows, columns = feature.coordinates.T
        image[rows, columns, 0] = 1.0
        image[rows, columns, 2] = 1.0  # magenta predictions; overlap is white
    return image


def centroid(feature) -> tuple[float, float]:
    row, column = feature.coordinates.mean(axis=0)
    return float(column), float(row)


def run_visualization(example_count: int = 4) -> Path:
    manifest, entries = load_all_configs()
    _, config = next(pair for pair in entries if pair[0].get("operating_points"))
    output_root = resolve_path(manifest["output_root"])
    metrics_root = output_root / "metrics"
    per_image = read_csv(metrics_root / "validation" / "per_image.csv")
    candidates = [row for row in per_image if row["model_id"] == config["id"]]
    selected = select_quantile_rows(candidates, example_count)
    selected_ids = {row["source_id"] for row in selected}

    dataset = resolve_path(config["dataset_root"])
    samples = [
        sample for sample in read_samples(dataset, "val")
        if Path(sample["image_name"]).stem in selected_ids
    ]
    records = {record["source_id"]: record for record in load_targets(dataset, samples)}
    threshold = float(json.loads((metrics_root / "thresholds.json").read_text())[config["id"]])
    cache = cache_directory(output_root / "cache", "validation", config["id"])

    figure, axes = plt.subplots(
        len(selected), 4, figsize=(14, 3.15 * len(selected)), squeeze=False,
        constrained_layout=True,
    )
    metric_rows = []
    for row_index, selected_row in enumerate(selected):
        source_id = selected_row["source_id"]
        record = records[source_id]
        probability = np.load(cache / f"{source_id}.npy", mmap_mode="r")
        prediction = (probability > threshold) & record["valid"]
        pixel = pixel_metrics(
            prediction,
            record["target"],
            record["valid"],
            target_band=record["target_band"],
            target_skeleton=record["target_skeleton"],
        )
        predicted_crests = extract_event_features(prediction)
        labeled_crests = record["target_crests"]
        scores = score_feature_matrix(predicted_crests, labeled_crests)
        matches = maximum_quality_assignment(scores)
        crest_counts = soft_crest_counts(
            prediction, target_features=labeled_crests
        )
        crest = ratios(crest_counts)
        pair_mean = (
            float(np.mean([scores[label, pred] for label, pred in matches]))
            if matches else float("nan")
        )

        axes[row_index, 0].imshow(record["target"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 0].set_title("Labeled crest mask")
        axes[row_index, 1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"Prediction (threshold {threshold:.2f})")
        axes[row_index, 2].imshow(
            pixel_error_image(prediction, record["target"], record["valid"])
        )
        axes[row_index, 2].set_title(
            f"Pixel: Dice {pixel['dice']:.3f}, IoU {pixel['iou_fg']:.3f}\n"
            f"bF1 {pixel['boundary_f1']:.3f}, clDice {pixel['cldice']:.3f}"
        )
        axes[row_index, 3].imshow(
            crest_image(predicted_crests, labeled_crests, prediction.shape)
        )
        for label_index, prediction_index in matches:
            pred_x, pred_y = centroid(predicted_crests[prediction_index])
            label_x, label_y = centroid(labeled_crests[label_index])
            axes[row_index, 3].plot(
                (label_x, pred_x), (label_y, pred_y), color="gold", alpha=0.6,
                linewidth=0.7,
            )
        axes[row_index, 3].set_title(
            f"Crests: P {crest['crest_precision']:.3f}, "
            f"R {crest['crest_recall']:.3f}, F1 {crest['crest_f1']:.3f}\n"
            f"{len(predicted_crests)} predicted / {len(labeled_crests)} labeled; "
            f"pair clDice {pair_mean:.3f}"
        )
        axes[row_index, 0].set_ylabel(source_id)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

        metric_rows.append({
            "source_id": source_id,
            "selection_crest_f1_quantile": row_index / max(1, len(selected) - 1),
            "threshold": threshold,
            **pixel,
            **crest_counts,
            **crest,
            "matched_pair_cldice_mean": pair_mean,
        })

    figure.legend(
        handles=[
            Patch(color=(0.20, 0.90, 0.20), label="pixel true positive"),
            Patch(color=(1.00, 0.20, 0.20), label="pixel false positive"),
            Patch(color=(0.20, 0.40, 1.00), label="pixel false negative"),
            Patch(color="cyan", label="labeled crest skeleton"),
            Patch(color="magenta", label="predicted crest skeleton"),
            Patch(color="gold", label="one-to-one matched pair"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )
    output = metrics_root / "examples"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "crest_and_pixel_metrics.png", dpi=180)
    figure.savefig(output / "crest_and_pixel_metrics.pdf")
    plt.close(figure)
    write_csv(output / "crest_and_pixel_metrics.csv", metric_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    output = run_visualization(args.examples)
    print(f"Wrote metric examples to {output}")


if __name__ == "__main__":
    main()
