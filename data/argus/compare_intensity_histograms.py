#!/usr/bin/env python3
"""Compare sampled-video intensities with the full-split and OOD images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from dropout_common import (
    REPO_ROOT,
    histogram_quantile,
    normalize_histograms,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO_HISTOGRAMS = HERE / "nonzero_pixel_intensity_histograms.npz"
DEFAULT_FULL_SPLIT = REPO_ROOT / "model" / "data" / "full_split"
DEFAULT_OOD_TEST = REPO_ROOT / "model" / "data" / "ood_test"
DEFAULT_SUMMARY = HERE / "intensity_histogram_reference_comparison.csv"
DEFAULT_SUPPORT = HERE / "intensity_histogram_support.csv"
DEFAULT_FIGURE = HERE / "intensity_histogram_reference_comparison.png"


def load_image_histograms(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with (root / "image_mask_mapping.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = []
    splits = []
    counts = []
    for row in rows:
        path = root / "images" / row["image_name"]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 3:
            if not np.all(image == image[..., :1]):
                raise ValueError(f"RGB channels differ in {path}")
            image = image[..., 0]
        values = image[image != 0]
        if values.size == 0:
            raise ValueError(f"no nonzero pixels in {path}")
        names.append(row["image_name"])
        splits.append(row["split"])
        counts.append(np.bincount(values, minlength=256))
    return np.asarray(names), np.asarray(splits), np.stack(counts)


def augmented_train_histograms(
    root: Path, patches_per_image: int, epoch: int, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    from common.normalization import denorm
    from dataset import WaveBreakingDataset

    dataset = WaveBreakingDataset(
        root,
        split="train",
        transform_size=224,
        patches_per_image=patches_per_image,
        augment=True,
        seed=42,
    )
    dataset.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    counts = np.zeros((len(dataset), 256), dtype=np.int64)
    offset = 0
    for images, _ in loader:
        grayscale = denorm(images).mean(dim=1).mul(255).round().to(torch.uint8)
        for index, values in enumerate(grayscale.numpy()):
            values = values[values != 0]
            counts[offset + index] = np.bincount(values, minlength=256)
        offset += images.shape[0]
    names = np.asarray(
        [
            f"{row['image_name']}#patch-{patch:02d}"
            for row in dataset.samples
            for patch in range(patches_per_image)
        ]
    )
    return names, counts


def write_figure(
    path: Path,
    video_pdf: np.ndarray,
    groups: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intensity = np.arange(256)
    colors = {
        "videos": "black",
        "train": "#0072B2",
        "val": "#E69F00",
        "test": "#009E73",
        "augmented_train": "#D55E00",
        "ood_test": "#CC79A7",
    }
    labels = {
        "videos": "sampled videos",
        "train": "train images",
        "val": "validation images",
        "test": "test images",
        "augmented_train": "augmented train patches",
        "ood_test": "OOD-test images",
    }
    support_titles = {
        "full_split": "Full-split support",
        "augmented_train": "Augmented-train support",
        "ood_test": "OOD-test support",
    }
    envelope_labels = {
        "full_split": "full-split image envelope",
        "augmented_train": "augmented train-patch envelope",
        "ood_test": "OOD-test image envelope",
    }
    pooled_video = normalize_histograms(video_pdf.sum(axis=0)[None])[0]

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    axes = axes.ravel()
    axes[0].plot(
        intensity[1:],
        pooled_video[1:],
        color=colors["videos"],
        lw=2.4,
        label=labels["videos"],
    )
    for name in ("train", "val", "test", "ood_test", "augmented_train"):
        pooled = normalize_histograms(groups[name].sum(axis=0)[None])[0]
        axes[0].plot(
            intensity[1:],
            pooled[1:],
            color=colors[name],
            lw=1.6,
            label=labels[name],
        )
    axes[0].set(
        xlim=(1, 255),
        xlabel="grayscale intensity (zero excluded)",
        ylabel="probability per intensity level",
        title="Pooled histograms",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    for axis, (name, image_pdf) in zip(
        axes[1:], references.items(), strict=True
    ):
        image_cdf = np.cumsum(image_pdf, axis=1)
        for cdf in np.cumsum(video_pdf, axis=1):
            axis.plot(intensity[1:], cdf[1:], color="0.35", alpha=0.12, lw=0.6)
        axis.fill_between(
            intensity[1:],
            image_cdf[:, 1:].min(axis=0),
            image_cdf[:, 1:].max(axis=0),
            color={
                "full_split": colors["train"],
                "augmented_train": colors["augmented_train"],
                "ood_test": colors["ood_test"],
            }[name],
            alpha=0.25,
            label=envelope_labels[name],
        )
        axis.plot(
            intensity[1:],
            np.cumsum(pooled_video)[1:],
            color="black",
            lw=2.0,
            label="pooled videos",
        )
        axis.set(
            xlim=(1, 255),
            ylim=(0, 1),
            xlabel="grayscale intensity (zero excluded)",
            ylabel="cumulative probability",
            title=support_titles[name],
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="lower right")

    figure.suptitle("Argus video intensities versus canonical dataset images")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-histograms", type=Path, default=DEFAULT_VIDEO_HISTOGRAMS)
    parser.add_argument("--full-split", type=Path, default=DEFAULT_FULL_SPLIT)
    parser.add_argument("--ood-test", type=Path, default=DEFAULT_OOD_TEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--augmented-patches-per-image", type=int, default=64)
    parser.add_argument("--augmentation-epoch", type=int, default=0)
    parser.add_argument("--augmentation-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.augmented_patches_per_image < 1 or args.augmentation_workers < 0:
        raise SystemExit("augmentation patches must be positive and workers nonnegative")
    with np.load(args.video_histograms) as archive:
        video_names = archive["video"].astype(str)
        video_counts = archive["counts"].astype(float)
    video_pdf = normalize_histograms(video_counts)
    pooled_video = normalize_histograms(video_counts.sum(axis=0)[None])[0]

    full_names, full_splits, full_counts = load_image_histograms(args.full_split)
    ood_names, _, ood_counts = load_image_histograms(args.ood_test)
    print(
        f"Sampling augmentation epoch {args.augmentation_epoch} at "
        f"{args.augmented_patches_per_image} patches per train image"
    )
    augmented_names, augmented_counts = augmented_train_histograms(
        args.full_split,
        args.augmented_patches_per_image,
        args.augmentation_epoch,
        args.augmentation_workers,
    )
    groups = {
        split: full_counts[full_splits == split]
        for split in ("train", "val", "test")
    }
    groups["full_split"] = full_counts
    groups["ood_test"] = ood_counts
    groups["augmented_train"] = augmented_counts

    intensity = np.arange(256)
    summary_rows = [
        {
            "reference_set": "sampled_videos",
            "histograms": video_counts.shape[0],
            "nonzero_pixels": int(video_counts.sum()),
            "mean_intensity": float(pooled_video @ intensity),
            "p05": histogram_quantile(pooled_video, 0.05),
            "median": histogram_quantile(pooled_video, 0.50),
            "p95": histogram_quantile(pooled_video, 0.95),
            "pooled_total_variation_to_videos": 0.0,
            "pooled_max_cdf_error_to_videos": 0.0,
            "pooled_wasserstein_intensity_levels": 0.0,
        }
    ]
    for name in (
        "train",
        "val",
        "test",
        "augmented_train",
        "full_split",
        "ood_test",
    ):
        counts = groups[name]
        pooled = normalize_histograms(counts.sum(axis=0)[None])[0]
        difference = pooled - pooled_video
        cdf_difference = np.cumsum(difference)
        summary_rows.append(
            {
                "reference_set": name,
                "histograms": counts.shape[0],
                "nonzero_pixels": int(counts.sum()),
                "mean_intensity": float(pooled @ intensity),
                "p05": histogram_quantile(pooled, 0.05),
                "median": histogram_quantile(pooled, 0.50),
                "p95": histogram_quantile(pooled, 0.95),
                "pooled_total_variation_to_videos": float(
                    0.5 * np.abs(difference).sum()
                ),
                "pooled_max_cdf_error_to_videos": float(
                    np.abs(cdf_difference).max()
                ),
                "pooled_wasserstein_intensity_levels": float(
                    np.abs(cdf_difference).sum()
                ),
            }
        )

    references = {
        "full_split": normalize_histograms(full_counts),
        "augmented_train": normalize_histograms(augmented_counts),
        "ood_test": normalize_histograms(ood_counts),
    }
    reference_names = {
        "full_split": full_names,
        "augmented_train": augmented_names,
        "ood_test": ood_names,
    }
    support_rows = []
    for reference, image_pdf in references.items():
        image_cdf = np.cumsum(image_pdf, axis=1)
        video_cdf = np.cumsum(video_pdf, axis=1)
        envelope_violation = np.maximum(image_cdf.min(axis=0) - video_cdf, 0)
        envelope_violation += np.maximum(video_cdf - image_cdf.max(axis=0), 0)

        tv_threshold: float | None = None
        cdf_threshold: float | None = None
        if reference != "augmented_train":
            internal_difference = image_pdf[:, None, :] - image_pdf[None, :, :]
            internal_tv = 0.5 * np.abs(internal_difference).sum(axis=2)
            internal_cdf = np.abs(
                np.cumsum(internal_difference, axis=2)
            ).max(axis=2)
            np.fill_diagonal(internal_tv, np.inf)
            np.fill_diagonal(internal_cdf, np.inf)
            tv_threshold = float(np.quantile(internal_tv.min(axis=1), 0.95))
            cdf_threshold = float(np.quantile(internal_cdf.min(axis=1), 0.95))

        for index, video in enumerate(video_names):
            difference = video_pdf[index] - image_pdf
            total_variation = 0.5 * np.abs(difference).sum(axis=1)
            max_cdf_error = np.abs(np.cumsum(difference, axis=1)).max(axis=1)
            nearest_tv = int(np.argmin(total_variation))
            nearest_cdf = int(np.argmin(max_cdf_error))
            supported = (
                total_variation[nearest_tv] <= tv_threshold
                and max_cdf_error[nearest_cdf] <= cdf_threshold
                if tv_threshold is not None and cdf_threshold is not None
                else None
            )
            support_rows.append(
                {
                    "reference_set": reference,
                    "video": video,
                    "nearest_image_by_tv": reference_names[reference][nearest_tv],
                    "nearest_total_variation": float(total_variation[nearest_tv]),
                    "nearest_image_by_cdf": reference_names[reference][nearest_cdf],
                    "nearest_max_cdf_error": float(max_cdf_error[nearest_cdf]),
                    "reference_p95_nearest_total_variation": tv_threshold,
                    "reference_p95_nearest_max_cdf_error": cdf_threshold,
                    "supported_by_both_p95_thresholds": supported,
                    "within_cdf_envelope": bool(
                        envelope_violation[index].max() == 0
                    ),
                    "cdf_envelope_max_excursion": float(
                        envelope_violation[index].max()
                    ),
                }
            )

    write_csv(args.summary, summary_rows)
    write_csv(args.support, support_rows)
    write_figure(args.figure, video_pdf, groups, references)

    print("\nPooled nonzero-intensity comparison")
    print("  set          hists  mean  p05  median  p95     TV    max CDF    W1")
    for row in summary_rows:
        print(
            f"  {row['reference_set']:<13} {row['histograms']:>6} "
            f"{row['mean_intensity']:>5.1f} {row['p05']:>4} {row['median']:>7} "
            f"{row['p95']:>4} {row['pooled_total_variation_to_videos']:>6.3f} "
            f"{row['pooled_max_cdf_error_to_videos']:>10.3f} "
            f"{row['pooled_wasserstein_intensity_levels']:>5.1f}"
        )
    for reference in references:
        rows = [row for row in support_rows if row["reference_set"] == reference]
        inside = sum(row["within_cdf_envelope"] for row in rows)
        if reference == "augmented_train":
            print(
                f"  {reference}: {inside}/{len(rows)} videos lie wholly inside "
                "the augmented-patch CDF envelope"
            )
        else:
            supported = sum(row["supported_by_both_p95_thresholds"] for row in rows)
            print(
                f"  {reference}: {supported}/{len(rows)} videos pass both "
                f"internal-p95 nearest-image thresholds; {inside}/{len(rows)} "
                "lie wholly inside the image CDF envelope"
            )
    print(f"\nSummary: {args.summary}")
    print(f"Per-video support: {args.support}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
