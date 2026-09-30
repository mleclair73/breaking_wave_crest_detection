#!/usr/bin/env python3
"""Build the unlabelled, date-disjoint 20-image OOD test set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


WIDTH, HEIGHT = 512, 500
# Fixed windows preserve two visually audited edge cases: rank 1 uses the
# visible middle-twilight interval rather than its nearly black complete run;
# rank 9 uses the late, higher-contrast D4 option selected during review.
START_BY_RANK = {1: 1280, 9: 1884}
SELECTION_MODE_BY_RANK = {1: "brightness_override", 9: "lighting_override"}
YFRFS_BY_RANK = {
    1: (300, 700, 1100),
    2: (300, 700, 1100),
    3: (300, 700, 1100),
    4: (300, 700, 1100),
    5: (400, 1000),
    6: (400, 1000),
    7: (400, 1000),
    8: (700,),
    9: (550,),
}
def quality(crop: np.ndarray) -> dict[str, int | float | bool]:
    """Apply the frozen v6 missing-data and glare gates to an RGB crop."""
    zero = np.all(crop == 0, axis=2)
    top_margin = 0
    for row in zero:
        if row.all():
            top_margin += 1
        else:
            break
    zero_outside = int(zero[top_margin:].sum())
    full_zero_time_columns = int(zero.all(axis=0).sum())
    saturated_fraction = float(np.all(crop >= 250, axis=2).mean())
    missing_fraction = float(zero.mean())
    accepted = (
        top_margin <= 78
        and zero_outside <= 2048
        and full_zero_time_columns == 0
        and saturated_fraction <= 0.25
    )
    return {
        "accepted": accepted,
        "top_no_data_margin_px": top_margin,
        "zero_pixels_outside_top_margin": zero_outside,
        "full_zero_time_columns": full_zero_time_columns,
        "missing_pixel_fraction": missing_fraction,
        "saturated_pixel_fraction": saturated_fraction,
    }


def decode_rows(video: Path, rows: list[int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = np.empty((frame_count, len(rows), HEIGHT, 3), dtype=np.uint8)
    index = 0
    while index < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1600, HEIGHT):
            raise ValueError(f"unexpected video geometry {frame.shape}: {video}")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        decoded[index] = frame[rows]
        index += 1
    cap.release()
    if index != frame_count:
        raise RuntimeError(f"decoded {index}/{frame_count} frames from {video}")
    return decoded


def candidate_starts(frame_count: int, row: dict[str, str], width: int) -> tuple[list[int], int]:
    run_start = int(row["longest_complete_start"])
    run_end = int(row["longest_complete_end"])
    preferred = max(0, min(frame_count - width, (run_start + run_end - width) // 2))
    starts = set(range(0, frame_count - width + 1, 32))
    starts.update({preferred, max(0, run_start), max(0, min(frame_count - width, run_end - width))})
    return sorted(starts), preferred


def choose_window(
    decoded: np.ndarray, row: dict[str, str], width: int,
) -> tuple[int, list[dict[str, int | float | bool]], str]:
    """Choose a shared window, preferring strict QA and otherwise least missingness."""
    starts, preferred = candidate_starts(decoded.shape[0], row, width)
    evaluated: list[tuple[int, list[dict[str, int | float | bool]]]] = []
    for start in starts:
        metrics = [quality(decoded[start:start + width, i].transpose(1, 0, 2)) for i in range(decoded.shape[1])]
        evaluated.append((start, metrics))

    strict = [item for item in evaluated if all(bool(metric["accepted"]) for metric in item[1])]
    if strict:
        start, metrics = min(strict, key=lambda item: abs(item[0] - preferred))
        return start, metrics, "strict_quality_pass"

    # Missing pixels are acceptable for this deliberately difficult OOD set, but
    # should never be preferred to a populated view. Keep the synchronized spatial
    # plan fixed and minimize the worst affected crop, followed by total missingness.
    # Saturation is ranked afterward because genuine near-horizon brightness is one
    # of the intended OOD conditions.
    def fallback_score(item: tuple[int, list[dict[str, int | float | bool]]]) -> tuple[object, ...]:
        start, metrics = item
        return (
            max(float(metric["missing_pixel_fraction"]) for metric in metrics),
            sum(float(metric["missing_pixel_fraction"]) for metric in metrics),
            max(int(metric["full_zero_time_columns"]) for metric in metrics),
            max(float(metric["saturated_pixel_fraction"]) for metric in metrics),
            abs(start - preferred),
        )

    start, metrics = min(evaluated, key=fallback_score)
    return start, metrics, "least_missing_fallback"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--current-conditions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidate_reader = csv.DictReader(stream)
        candidate_fieldnames = candidate_reader.fieldnames
        candidates = sorted(
            (row for row in candidate_reader if int(row["target_crops"]) > 0),
            key=lambda row: int(row["rank"]),
        )
    if not candidate_fieldnames:
        raise ValueError(f"candidate CSV has no header: {args.candidate_csv}")
    if sum(int(row["target_crops"]) for row in candidates) != 20:
        raise ValueError("candidate plan must request exactly 20 crops")
    current_dates = set(pd.to_datetime(pd.read_csv(args.current_conditions)["acquisition_utc"], utc=True).dt.date)
    candidate_dates = set(pd.to_datetime([row["acquisition_utc"] for row in candidates], utc=True).date)
    if current_dates & candidate_dates:
        raise ValueError(f"OOD date leakage: {sorted(current_dates & candidate_dates)}")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    image_index = 1
    source_hashes: dict[str, str] = {}
    for source in candidates:
        rank = int(source["rank"])
        count = int(source["target_crops"])
        width = WIDTH
        video = args.video_dir / source["video_name"]
        if not video.is_file():
            raise FileNotFoundError(video)
        source_hashes[video.name] = sha256(video)
        yfrfs = YFRFS_BY_RANK[rank]
        if len(yfrfs) != count:
            raise ValueError(f"rank {rank}: yFRF plan does not match target_crops")

        rows = [1500 - yfrf for yfrf in yfrfs]
        decoded = decode_rows(video, rows)
        if rank in START_BY_RANK:
            start = START_BY_RANK[rank]
            if not 0 <= start <= decoded.shape[0] - width:
                raise ValueError(f"rank {rank}: fixed start {start} is outside the video")
            metrics = [
                quality(decoded[start:start + width, i].transpose(1, 0, 2))
                for i in range(decoded.shape[1])
            ]
            selection_mode = SELECTION_MODE_BY_RANK[rank]
        else:
            start, metrics, selection_mode = choose_window(decoded, source, width)

        for row_index, (yfrf, metric) in enumerate(zip(yfrfs, metrics)):
            pixel_row = 1500 - yfrf
            source_crop = decoded[start:start + width, row_index].transpose(1, 0, 2)
            crop = np.ascontiguousarray(np.flipud(source_crop))
            if crop.shape != (HEIGHT, width, 3):
                raise ValueError(f"unexpected crop shape {crop.shape}")
            image_name = f"ood_test_{image_index:03d}.png"
            Image.fromarray(crop, "RGB").save(images_dir / image_name)
            intensity = crop.mean(axis=2)
            records.append({
                "sample_id": Path(image_name).stem,
                "image_name": image_name,
                "product": "ood_test",
                "split": "ood_test",
                "source_video": f"data/video_dataset/argus/{video.name}",
                "source_rank": rank,
                "source_priority": source["priority"],
                "acquisition_utc": source["acquisition_utc"],
                "yfrf": yfrf,
                "pixel_row": pixel_row,
                "x_start": 0,
                "x_end": HEIGHT,
                "t_start": start,
                "t_end": start + width,
                "crop_width": width,
                "crop_height": HEIGHT,
                "top_no_data_margin_px": metric["top_no_data_margin_px"],
                "zero_pixels_outside_top_margin": metric["zero_pixels_outside_top_margin"],
                "full_zero_time_columns": metric["full_zero_time_columns"],
                "missing_pixel_fraction": f"{float(metric['missing_pixel_fraction']):.6f}",
                "saturated_pixel_fraction": f"{float(metric['saturated_pixel_fraction']):.6f}",
                "selection_mode": selection_mode,
                "water_level_m": source["water_level_m"],
                "Hs_m": source["Hs_m"],
                "Tp_s": source["Tp_s"],
                "peak_frequency_hz": source["peak_frequency_hz"],
                "mean_direction_deg": source["mean_direction_deg"],
                "wind_speed_m_s": source["wind_speed_m_s"],
                "solar_elevation_deg": source["solar_elevation_deg"],
                "solar_azimuth_deg": source["solar_azimuth_deg"],
                "joint_distance_sigma": source["joint_distance_sigma"],
                "selection_rationale": source["rationale"],
                "intensity_p01": f"{np.quantile(intensity, 0.01):.2f}",
                "intensity_p50": f"{np.quantile(intensity, 0.50):.2f}",
                "intensity_p99": f"{np.quantile(intensity, 0.99):.2f}",
                "orientation": "flipud(source_timestack) -> model/CVAT orientation",
            })
            image_index += 1

    if len(records) != 20:
        raise ValueError(f"built {len(records)} images, expected 20")
    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with (args.output_dir / "source_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    figure, axes = plt.subplots(4, 5, figsize=(15, 11.5), squeeze=False)
    for axis, record in zip(axes.flat, records):
        axis.imshow(Image.open(images_dir / str(record["image_name"])))
        axis.set_title(
            f"{record['sample_id']} · P{record['source_rank']} · yFRF {record['yfrf']}\n"
            f"Hs {record['Hs_m']} m · Tp {record['Tp_s']} s · WL {record['water_level_m']} m",
            fontsize=7,
        )
        axis.axis("off")
    figure.suptitle("Unlabelled date-disjoint OOD test set (model/CVAT orientation, n=20)")
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(args.output_dir / "contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    verification = {
        "image_count": len(records),
        "source_video_count": len({record["source_video"] for record in records}),
        "source_date_count": len(candidate_dates),
        "current_dataset_date_overlap": [],
        "geometry": {"all_images": [WIDTH, HEIGHT]},
        "orientation": "RGB source rows -> (cross_shore,time,RGB) -> flipud once",
        "strict_quality_pass_count": sum(record["selection_mode"] == "strict_quality_pass" for record in records),
        "least_missing_fallback_count": sum(record["selection_mode"] == "least_missing_fallback" for record in records),
        "brightness_override_count": sum(record["selection_mode"] == "brightness_override" for record in records),
        "lighting_override_count": sum(record["selection_mode"] == "lighting_override" for record in records),
        "selection_mode_counts": {
            mode: sum(record["selection_mode"] == mode for record in records)
            for mode in sorted({str(record["selection_mode"]) for record in records})
        },
        "max_missing_pixel_fraction": max(float(record["missing_pixel_fraction"]) for record in records),
        "crops_with_full_zero_time_columns": sum(int(record["full_zero_time_columns"]) > 0 for record in records),
        "quality_gates": {
            "top_no_data_margin_px_max": 78,
            "zero_pixels_outside_top_margin_max": 2048,
            "full_zero_time_columns": 0,
            "saturated_pixel_fraction_max": 0.25,
        },
        "missing_data_policy": (
            "Prefer a synchronized window passing the frozen v6 gates; when none exists, "
            "retain the fixed 500x512 geometry and planned yFRF rows and select the window "
            "that minimizes worst-crop then total missing-pixel fraction before saturation. "
            "This preserves real near-horizon brightness rather than selecting blank frames."
        ),
        "source_video_sha256": source_hashes,
    }
    (args.output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    readme = """# Date-disjoint OOD test set

This is a fixed, human-reviewed 20-image OOD evaluation set. It contains 20
500x512 RGB timestacks from nine source videos on eight dates. Those dates do
not occur in the combined training, validation, or regular test splits.

Use this set only after model and decision-threshold selection. Do not move its
images or labels into training/validation or use OOD results to tune a model.

## Contents

- `images/`: the 20 model-oriented PNG inputs.
- `manifest.csv`: exact source video, spatial row, frame interval, conditions,
  missing-data metrics, orientation transform, and source selection rank.
- `source_candidates.csv`: the exact nine-source, 20-crop input plan.
- `verification.json`: leakage, geometry, QA-policy, and source-AVI hashes.
- `contact_sheet.png`: visual orientation and content audit.
- `build_ood_test.py`: deterministic regeneration code.
- `cvat_job_22_export_20260918.zip`: preserved reviewed CVAT for images 1.1
  export.
- `annotations_human_canonical.xml`: manifest-ordered reviewed annotations,
  tagged `done` and `annotated` in model/CVAT orientation.
- `masks/` and `image_mask_mapping.csv`: one-pixel binary masks and the
  evaluation-loader mapping (`split=ood_test`).
- `annotation_status.json`: annotation provenance, hashes, counts, and per-image
  line totals.
- `annotation_overlay.png`: visual audit of the reviewed masks over all images.
- `import_reviewed_annotations.py`: validates the preserved export and
  deterministically regenerates canonical XML, masks, mapping, and status.
- `annotations_preseed.xml`, `ood_test_preannotations.zip`,
  `preannotation_overlay.png`, and `preannotation_summary.json`: original model
  suggestions retained for provenance; these are not ground truth.

All images use the same 500x512 geometry as the main dataset. The builder first
looks for a synchronized window that passes the frozen v6 gates. Where that is
impossible, it preserves the planned spatial rows and full width and chooses the
least-missing window. Rank 1 is explicitly fixed to middle-twilight frames
1280--1791 because its metadata-complete interval occurs after the sunset scene
has become nearly black. This retains limited missing columns instead of either
the heavily gapped early interval or unusable dark imagery. `selection_mode`
and all missingness values are explicit in `manifest.csv`. Rank 9 uses the
reviewed D4 option at yFRF 550 and frames 1884--2395 for better late-interval
contrast while retaining a strict quality pass.

## Regenerate images

From this directory (`model/data/ood_test`):

```bash
MPLCONFIGDIR=/tmp/mpl-ood-test python build_ood_test.py \\
  --video-dir ../../../data/argus \\
  --candidate-csv source_candidates.csv \\
  --current-conditions ../full_split/figures/acquisition_conditions.csv \\
  --output-dir .
```

The nine selected source AVIs must be present in `data/argus`. Their expected SHA-256
digests are recorded in `verification.json`. Rebuilding images leaves reviewed
label artifacts in place, but rewrites this README and the image-source
verification file.

## Regenerate reviewed labels

After the images exist, run:

```bash
python import_reviewed_annotations.py \\
  --archive cvat_job_22_export_20260918.zip \\
  --dataset .
```

This validates all 20 image identities, 512x500 geometry, `breaker` polylines,
and in-bounds vertices before writing labels. It preserves the single vertical
flip already applied during image extraction; annotation import applies no
additional orientation transform.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    script_target = args.output_dir / "build_ood_test.py"
    if Path(__file__).resolve() != script_target.resolve():
        script_target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"built {len(records)} OOD images from {len(source_hashes)} sources at {args.output_dir}")


if __name__ == "__main__":
    main()
