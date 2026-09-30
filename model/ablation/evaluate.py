#!/usr/bin/env python3
"""Evaluate cached complete-image predictions with one metric implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from .config import load_all_configs, resolve_path
from .infer import read_samples
from metrics.crest import (
    extract_event_features,
    metadata as crest_metric_metadata,
    ratios,
    soft_crest_counts,
)
from metrics.pixel import metrics as pixel_metrics


THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0
PIXEL_SUMMARY_METRICS = ("dice", "iou_fg", "boundary_f1", "cldice")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def load_targets(dataset: Path, samples: list[dict[str, str]]) -> list[dict]:
    records = []
    for sample in samples:
        image = np.asarray(
            Image.open(dataset / "images" / sample["image_name"]).convert("RGB")
        )
        valid = image.sum(axis=2) > 0
        target = np.asarray(
            Image.open(dataset / "masks" / sample["mask_name"]).convert("L")
        ) > 0
        target &= valid
        records.append({
            "sample": sample,
            "source_id": Path(sample["image_name"]).stem,
            "target": target,
            "valid": valid,
            "target_band": binary_dilation(target, iterations=2),
            "target_skeleton": skeletonize(target),
            "target_crests": extract_event_features(target),
        })
    return records


def cache_directory(cache_root: Path, condition: str, model_id: str) -> Path:
    path = cache_root / condition / model_id
    metadata = path / "metadata.json"
    if not metadata.is_file() or json.loads(metadata.read_text()).get("status") != "complete":
        raise FileNotFoundError(f"Incomplete inference cache: {path}")
    return path


def threshold_histograms(
    cache: Path, targets: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros(len(THRESHOLDS) + 1, dtype=np.int64)
    negative = np.zeros_like(positive)
    for record in targets:
        probability = np.load(cache / f"{record['source_id']}.npy", mmap_mode="r")
        values = np.asarray(probability[record["valid"]], dtype=np.float32)
        target_values = record["target"][record["valid"]]
        buckets = np.searchsorted(THRESHOLDS, values, side="left")
        positive += np.bincount(
            buckets[target_values], minlength=len(THRESHOLDS) + 1
        )
        negative += np.bincount(
            buckets[~target_values], minlength=len(THRESHOLDS) + 1
        )
    return positive, negative


def threshold_curve(positive: np.ndarray, negative: np.ndarray) -> list[dict]:
    positive_predicted = np.flip(np.cumsum(np.flip(positive)))[1:]
    negative_predicted = np.flip(np.cumsum(np.flip(negative)))[1:]
    positive_total = int(positive.sum())
    rows = []
    for index, threshold in enumerate(THRESHOLDS):
        tp = int(positive_predicted[index])
        fp = int(negative_predicted[index])
        fn = positive_total - tp
        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
            "iou_fg": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        })
    return rows


def select_threshold(rows: list[dict]) -> float:
    return max(
        rows,
        key=lambda row: (
            row["dice"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )["threshold"]


def calibrate_thresholds(
    entries: list[tuple[dict, dict]],
    cache_root: Path,
    targets: list[dict],
    output_root: Path,
) -> dict[str, float]:
    thresholds = {}
    rows = []
    curve_dir = output_root / "metrics" / "threshold_calibration"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for _, config in entries:
        cache = cache_directory(cache_root, "validation", config["id"])
        curve = threshold_curve(*threshold_histograms(cache, targets))
        threshold = select_threshold(curve)
        thresholds[config["id"]] = threshold
        selected = next(row for row in curve if row["threshold"] == threshold)
        rows.append({
            "model_id": config["id"],
            "model": config["label"],
            "threshold": threshold,
            "selection_metric": "complete_image_dataset_dice",
            "dice": selected["dice"],
            "iou_fg": selected["iou_fg"],
        })
        write_csv(curve_dir / f"{config['id']}.csv", curve)
    write_csv(output_root / "metrics" / "thresholds.csv", rows)
    (output_root / "metrics" / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n"
    )
    return thresholds


def evaluate_source(
    cache: Path,
    record: dict,
    threshold: float,
    config: dict,
    condition: str,
) -> dict:
    probability = np.load(cache / f"{record['source_id']}.npy", mmap_mode="r")
    prediction = (probability > threshold) & record["valid"]
    pixel = pixel_metrics(
        prediction,
        record["target"],
        record["valid"],
        target_band=record["target_band"],
        target_skeleton=record["target_skeleton"],
    )
    crest = soft_crest_counts(
        prediction, target_features=record["target_crests"]
    )
    return {
        "condition": condition,
        "model_id": config["id"],
        "model": config["label"],
        "source_id": record["source_id"],
        "threshold": threshold,
        "inference": "complete_image_hanning_tiled",
        **pixel,
        **crest,
        **ratios(crest),
    }


def summarize(config: dict, condition: str, rows: list[dict]) -> dict:
    crest_counts = {
        "crest_predicted": sum(int(row["crest_predicted"]) for row in rows),
        "crest_labeled": sum(int(row["crest_labeled"]) for row in rows),
        "crest_quality_sum": sum(float(row["crest_quality_sum"]) for row in rows),
    }
    result = {
        "condition": condition,
        "model_id": config["id"],
        "model": config["label"],
        "threshold": rows[0]["threshold"],
        "n_images": len(rows),
    }
    for metric in PIXEL_SUMMARY_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = float(np.nanmean(values))
        result[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
    result.update(crest_counts)
    result.update(ratios(crest_counts))
    return result


def evaluate_condition(
    entries: list[tuple[dict, dict]],
    condition: str,
    dataset: Path,
    samples: list[dict[str, str]],
    cache_root: Path,
    output_root: Path,
    thresholds: dict[str, float],
    workers: int,
    targets: list[dict] | None = None,
) -> None:
    targets = targets if targets is not None else load_targets(dataset, samples)
    all_rows = []
    summaries = []
    for _, config in entries:
        cache = cache_directory(cache_root, condition, config["id"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(
                lambda record: evaluate_source(
                    cache, record, thresholds[config["id"]], config, condition
                ),
                targets,
            ))
        all_rows.extend(rows)
        summaries.append(summarize(config, condition, rows))
    output = output_root / "metrics" / condition
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_image.csv", all_rows)
    write_csv(output / "summary.csv", summaries)
    (output / "summary.json").write_text(
        json.dumps(json_safe(summaries), indent=2, allow_nan=False) + "\n"
    )


def run_evaluation(
    workers: int = 8,
    manifest_path: str | Path | None = None,
    conditions: tuple[str, ...] = ("validation", "missing_data", "ood"),
    thresholds_from: str | Path | None = None,
) -> None:
    manifest, entries = (
        load_all_configs(manifest_path)
        if manifest_path is not None
        else load_all_configs()
    )
    output_root = resolve_path(manifest["output_root"])
    cache_root = output_root / "cache"
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics" / "crest_metric_definition.json").write_text(
        json.dumps(crest_metric_metadata(), indent=2) + "\n"
    )
    validation_dataset = resolve_path(entries[0][1]["dataset_root"])
    validation_samples = read_samples(validation_dataset, "val")
    validation_targets = load_targets(validation_dataset, validation_samples)
    if thresholds_from is None:
        thresholds = calibrate_thresholds(
            entries, cache_root, validation_targets, output_root
        )
    else:
        thresholds = {
            key: float(value)
            for key, value in json.loads(
                resolve_path(thresholds_from).read_text()
            ).items()
        }
        missing_thresholds = {
            config["id"] for _, config in entries
        } - set(thresholds)
        if missing_thresholds:
            raise ValueError(
                f"Missing frozen thresholds for {sorted(missing_thresholds)}"
            )
    if "validation" in conditions:
        evaluate_condition(
            entries,
            "validation",
            validation_dataset,
            validation_samples,
            cache_root,
            output_root,
            thresholds,
            workers,
            validation_targets,
        )
    if "missing_data" in conditions:
        missing_entries = [pair for pair in entries if pair[0].get("missing_data")]
        if missing_entries:
            evaluate_condition(
                missing_entries,
                "missing_data",
                validation_dataset,
                validation_samples,
                cache_root,
                output_root,
                thresholds,
                workers,
                validation_targets,
            )
    if "ood" in conditions:
        ood_entries = [pair for pair in entries if pair[0].get("ood")]
        if ood_entries:
            ood_dataset = resolve_path(entries[0][1]["ood_dataset_root"])
            evaluate_condition(
                ood_entries,
                "ood",
                ood_dataset,
                read_samples(ood_dataset, None),
                cache_root,
                output_root,
                thresholds,
                workers,
            )
    if "test" in conditions:
        test_entries = [
            pair for pair in entries if pair[0].get("holdout_test")
        ]
        if test_entries:
            test_samples = read_samples(validation_dataset, "test")
            evaluate_condition(
                test_entries,
                "test",
                validation_dataset,
                test_samples,
                cache_root,
                output_root,
                thresholds,
                workers,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest")
    parser.add_argument(
        "--thresholds-from",
        help="JSON mapping of model ids to frozen validation thresholds",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=("validation", "missing_data", "ood", "test"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run_evaluation(
        args.workers,
        manifest_path=args.manifest,
        conditions=tuple(
            args.condition or ("validation", "missing_data", "ood")
        ),
        thresholds_from=args.thresholds_from,
    )


if __name__ == "__main__":
    main()
