#!/usr/bin/env python3
"""Cache complete-image Hanning-tiled probabilities for evaluation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from numpy.random import RandomState
from PIL import Image
from tqdm import tqdm

from checkpoint import load_checkpoint_model, seed_inference
from .checkpoint import latest_complete_run
from .config import MODEL_ROOT, load_all_configs, resolve_path
from augmentations import apply_missing_data_transform
from tiling import predict_tiled


def read_samples(dataset: Path, split: str | None) -> list[dict[str, str]]:
    with (dataset / "image_mask_mapping.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    if not rows:
        raise ValueError(f"No samples found in {dataset} for split={split!r}")
    return rows


def corrupt_missing_data(image: np.ndarray, seed: int, index: int) -> np.ndarray:
    tensor = torch.from_numpy(np.moveaxis(image, -1, 0).copy())
    valid = tensor.sum(dim=0, keepdim=True) > 0
    transformed = apply_missing_data_transform(
        tensor,
        RandomState(seed + 1_000_003 + index),
        p=1.0,
    )
    return np.moveaxis((transformed * valid).numpy(), 0, -1)


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def cache_model(
    config: dict,
    dataset: Path,
    samples: list[dict[str, str]],
    condition: str,
    cache_root: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    run = latest_complete_run(config)
    model, checkpoint_config = load_checkpoint_model(run, device)
    seed = int(checkpoint_config["seed"])
    seed_inference(seed, device)
    output = cache_root / condition / config["id"]
    output.mkdir(parents=True, exist_ok=True)
    patch_size = int(checkpoint_config["image_size"])
    # Tiling is an evaluation policy, not a learned model parameter. Read it
    # from the current study config so changing the policy does not require
    # retraining or mutating an existing checkpoint's frozen training config.
    overlap = int(config["tile_overlap"])
    metadata = {
        "status": "complete",
        "condition": condition,
        "model_id": config["id"],
        "model": config["label"],
        "run": str(run.relative_to(MODEL_ROOT)),
        "dataset": str(dataset.relative_to(MODEL_ROOT)),
        "samples": [sample["image_name"] for sample in samples],
        "inference": "complete_image_hanning_tiled",
        "patch_size": patch_size,
        "overlap": overlap,
        "dtype": "float32",
    }
    metadata_path = output / "metadata.json"
    try:
        existing_metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing_metadata = None
    reuse_cache = existing_metadata == metadata
    if not reuse_cache:
        in_progress = dict(metadata)
        in_progress["status"] = "in_progress"
        metadata_path.write_text(json.dumps(in_progress, indent=2) + "\n")
    for index, sample in enumerate(
        tqdm(samples, desc=f"{condition} {config['label']}")
    ):
        destination = output / f"{Path(sample['image_name']).stem}.npy"
        if reuse_cache and destination.is_file():
            continue
        image = np.asarray(
            Image.open(dataset / "images" / sample["image_name"]).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        inference_image = (
            corrupt_missing_data(image, seed, index)
            if condition == "missing_data"
            else image
        )
        probability = predict_tiled(
            model,
            inference_image,
            patch_size=patch_size,
            overlap_y=overlap,
            overlap_x=overlap,
            batch_size=batch_size,
            device=device,
        )
        atomic_save_array(destination, probability)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_inference(
    device: torch.device,
    batch_size: int = 32,
    conditions: tuple[str, ...] = ("validation", "missing_data", "ood"),
    manifest_path: str | Path | None = None,
) -> None:
    manifest, entries = (
        load_all_configs(manifest_path)
        if manifest_path is not None
        else load_all_configs()
    )
    cache_root = resolve_path(manifest["output_root"]) / "cache"
    for condition in conditions:
        if condition == "validation":
            selected = entries
            dataset = resolve_path(entries[0][1]["dataset_root"])
            samples = read_samples(dataset, "val")
        elif condition == "missing_data":
            selected = [(entry, config) for entry, config in entries if entry.get("missing_data")]
            dataset = resolve_path(entries[0][1]["dataset_root"])
            samples = read_samples(dataset, "val")
        elif condition == "ood":
            selected = [(entry, config) for entry, config in entries if entry.get("ood")]
            dataset = resolve_path(entries[0][1]["ood_dataset_root"])
            samples = read_samples(dataset, None)
        elif condition == "test":
            selected = [
                (entry, config)
                for entry, config in entries
                if entry.get("holdout_test")
            ]
            dataset = resolve_path(entries[0][1]["dataset_root"])
            samples = read_samples(dataset, "test")
        else:
            raise ValueError(f"Unknown inference condition: {condition}")
        for _, config in selected:
            cache_model(
                config, dataset, samples, condition, cache_root, device, batch_size
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--manifest")
    parser.add_argument(
        "--condition",
        action="append",
        choices=("validation", "missing_data", "ood", "test"),
    )
    args = parser.parse_args()
    run_inference(
        torch.device(args.device),
        batch_size=args.batch_size,
        conditions=tuple(args.condition or ("validation", "missing_data", "ood")),
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
