#!/usr/bin/env python3
"""Verify and fingerprint the frozen full_split segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FINGERPRINT_INPUTS = (
    "annotations_canonical.xml",
    "image_mask_mapping.csv",
    "split_assignment.csv",
    "split_manifest.csv",
)
EXPECTED_SPLITS = {"train": 172, "val": 43, "test": 41}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(root: Path, rows: list[dict[str, str]]) -> tuple[str, dict[str, str]]:
    paths = [root / name for name in FINGERPRINT_INPUTS]
    paths += [root / "images" / row["image_name"] for row in rows]
    paths += [root / "masks" / row["mask_name"] for row in rows]
    file_hashes = {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}
    digest = hashlib.sha256()
    for relative, file_digest in file_hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), file_hashes


def verify(root: Path, check_loader: bool) -> dict:
    rows = read_rows(root / "image_mask_mapping.csv")
    manifest = read_rows(root / "split_manifest.csv")
    assignments = read_rows(root / "split_assignment.csv")
    if len(rows) != 256 or len(manifest) != len(rows):
        raise ValueError(f"Expected 256 mapping/manifest rows, found {len(rows)}/{len(manifest)}")

    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")

    image_names = {path.name for path in (root / "images").glob("*.png")}
    mask_names = {path.name for path in (root / "masks").glob("*.png")}
    mapped_images = {row["image_name"] for row in rows}
    mapped_masks = {row["mask_name"] for row in rows}
    if len(mapped_images) != len(rows) or len(mapped_masks) != len(rows):
        raise ValueError("Mapping contains duplicate image or mask names")
    if image_names != mapped_images or mask_names != mapped_masks:
        raise ValueError(
            "Image/mask inventory differs from mapping: "
            f"images missing={sorted(mapped_images - image_names)[:3]}, extra={sorted(image_names - mapped_images)[:3]}; "
            f"masks missing={sorted(mapped_masks - mask_names)[:3]}, extra={sorted(mask_names - mapped_masks)[:3]}"
        )

    source_splits: dict[str, set[str]] = defaultdict(set)
    manifest_by_image = {row["image_name"]: row for row in manifest}
    geometries = set()
    mask_values = set()
    for row in rows:
        record = manifest_by_image.get(row["image_name"])
        if record is None or record["split"] != row["split"]:
            raise ValueError(f"Manifest mismatch for {row['image_name']}")
        source_splits[record["source_video"]].add(row["split"])
        with Image.open(root / "images" / row["image_name"]) as image, Image.open(
            root / "masks" / row["mask_name"]
        ) as mask:
            if image.size != mask.size:
                raise ValueError(f"Geometry mismatch for {row['image_name']}: {image.size} != {mask.size}")
            geometries.add(image.size)
            mask_values.update(np.unique(np.asarray(mask)).tolist())
    leaked = {source: sorted(splits) for source, splits in source_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"Source leakage across splits: {leaked}")
    if geometries != {(512, 500)}:
        raise ValueError(f"Unexpected geometries: {sorted(geometries)}")
    if not mask_values.issubset({0, 255}):
        raise ValueError(f"Masks are not binary: {sorted(mask_values)[:10]}")

    assignment_map = {row["source_id"]: row["split"] for row in assignments}
    if set(assignment_map) != set(source_splits):
        raise ValueError("split_assignment.csv does not contain exactly the manifest sources")
    for source, splits in source_splits.items():
        if assignment_map[source] != next(iter(splits)):
            raise ValueError(f"Split assignment mismatch for {source}")

    loader_checks = {}
    if check_loader:
        from dataset import WaveBreakingDataset

        for split, expected_sources in EXPECTED_SPLITS.items():
            dataset = WaveBreakingDataset(
                root, split=split, transform_size=224, patches_per_image=64,
                augment=False, seed=42,
            )
            if len(dataset.samples) != expected_sources or len(dataset) != expected_sources * 64:
                raise ValueError(f"Unexpected loader length for {split}: {len(dataset.samples)}/{len(dataset)}")
            # Exercise both edge patch indices for every source without decoding all 64 copies.
            for source_index in range(expected_sources):
                for patch_index in (0, 63):
                    image, mask = dataset[source_index * 64 + patch_index]
                    if tuple(image.shape) != (3, 224, 224) or tuple(mask.shape) != (1, 224, 224):
                        raise ValueError(f"Unexpected patch shape in {split} source {source_index}")
                    if not bool(image.isfinite().all()) or not bool(mask.isfinite().all()):
                        raise ValueError(f"Non-finite patch in {split} source {source_index}")
            loader_checks[split] = {
                "sources": expected_sources,
                "patches_per_source": 64,
                "total_patches": len(dataset),
                "checked_patch_indices_per_source": [0, 63],
                "shape": [3, 224, 224],
                "finite": True,
            }

    fingerprint, file_hashes = dataset_fingerprint(root, rows)
    return {
        "schema_version": 1,
        "dataset_version": "full_split-2026-09-16-revised",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "1de72f7",
        "dataset_sha256": fingerprint,
        "counts": {
            "images": len(image_names),
            "masks": len(mask_names),
            "source_videos": len(source_splits),
            "splits": dict(split_counts),
        },
        "geometry_wh": [512, 500],
        "mask_values": sorted(mask_values),
        "pairing": {"one_to_one": True, "orphan_images": 0, "orphan_masks": 0},
        "leakage": {"unit": "source_video", "cross_split_sources": []},
        "loader": loader_checks,
        "file_sha256": file_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-loader", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "verification.json"
    result = verify(root, check_loader=not args.skip_loader)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset verification passed: {result['dataset_sha256']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
