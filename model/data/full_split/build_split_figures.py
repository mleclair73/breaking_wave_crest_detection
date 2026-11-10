#!/usr/bin/env python3
"""Regenerate contact sheets and condition-coverage figures for full_split."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
SOURCE_CONDITIONS = FIGURES / "conditions_input.csv"
SPLITS = ("train", "val", "test")
COLORS = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
EXPECTED_COUNTS = {"train": 172, "val": 43, "test": 41}


def contact_sheets(mapping: list[dict[str, str]]) -> None:
    for split in SPLITS:
        chosen = sorted((row for row in mapping if row["split"] == split), key=lambda row: row["image_name"])
        columns = 8
        rows = int(np.ceil(len(chosen) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(16, 2.15 * rows), squeeze=False)
        for axis, row in zip(axes.flat, chosen):
            axis.imshow(Image.open(HERE / "images" / row["image_name"]))
            axis.set_title(f"{Path(row['image_name']).stem}\n{row['num_polylines']} crests", fontsize=6.5)
            axis.axis("off")
        for axis in axes.flat[len(chosen):]:
            axis.axis("off")
        figure.suptitle(f"Combined {split} split (n={len(chosen)})", fontsize=14, y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        figure.savefig(FIGURES / f"{split}_contact_sheet.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def acquisition_conditions(manifest: list[dict[str, str]]) -> pd.DataFrame:
    """Load the frozen, fully joined acquisition table bundled with the dataset."""
    frame = pd.read_csv(SOURCE_CONDITIONS)
    by_source: dict[str, str] = {}
    for row in manifest:
        if row["source_video"] in by_source and by_source[row["source_video"]] != row["split"]:
            raise ValueError(f"source crosses splits: {row['source_video']}")
        by_source[row["source_video"]] = row["split"]
    table_sources = set(frame["source_video"])
    if table_sources != set(by_source):
        raise ValueError(
            f"frozen condition table mismatch: missing={sorted(set(by_source)-table_sources)}, "
            f"extra={sorted(table_sources-set(by_source))}"
        )
    # Conditions are immutable source observations; the active assignment lives
    # in split_manifest.csv and may intentionally differ from the historical one.
    frame["split"] = frame["source_video"].map(by_source)
    return frame.sort_values(["split", "acquisition_utc"]).reset_index(drop=True)


def split_coverage(manifest: list[dict[str, str]]) -> None:
    acquisitions = sorted({row["acquisition_utc"] for row in manifest})
    yfrfs = sorted({int(float(row["yfrf"])) for row in manifest})
    figure, axes = plt.subplots(1, 2, figsize=(17, 5))
    width = 0.25
    x_acq = np.arange(len(acquisitions))
    x_yfrf = np.arange(len(yfrfs))
    for offset, split in enumerate(SPLITS):
        acquisition_counts = Counter(row["acquisition_utc"] for row in manifest if row["split"] == split)
        yfrf_counts = Counter(int(float(row["yfrf"])) for row in manifest if row["split"] == split)
        axes[0].bar(x_acq + (offset - 1) * width, [acquisition_counts[x] for x in acquisitions], width, label=split, color=COLORS[split])
        axes[1].bar(x_yfrf + (offset - 1) * width, [yfrf_counts[x] for x in yfrfs], width, label=split, color=COLORS[split])
    axes[0].set_xticks(x_acq, [x[5:16] for x in acquisitions], rotation=55, ha="right", fontsize=7)
    axes[0].set_title("Samples by acquisition")
    axes[1].set_xticks(x_yfrf, [str(x) for x in yfrfs], rotation=45, ha="right")
    axes[1].set_title("Samples by yFRF")
    axes[1].set_xlabel("yFRF (m)")
    for axis in axes:
        axis.set_ylabel("crops")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
    counts = Counter(row["split"] for row in manifest)
    figure.suptitle(
        "Combined split coverage — "
        f"{sum(counts.values())} crops ({counts['train']} train / {counts['val']} val / {counts['test']} test)"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "split_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def condition_plot(frame: pd.DataFrame, columns: list[str], shape: tuple[int, int], output: str, title: str) -> None:
    figure, axes = plt.subplots(*shape, figsize=(4 * shape[1], 3.6 * shape[0]), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        for index, split in enumerate(SPLITS):
            values = frame.loc[frame.split == split, column].dropna()
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
            axis.scatter(index + jitter, values, color=COLORS[split], s=38, alpha=0.9)
        axis.set_xticks(range(3), SPLITS)
        axis.set_ylabel(column)
        axis.grid(axis="y", alpha=0.25)
        if column.endswith("cloud cover"):
            axis.set_ylim(0, 1)
    for axis in axes.flat[len(columns):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(FIGURES / output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    with (HERE / "image_mask_mapping.csv").open(newline="", encoding="utf-8") as stream:
        mapping = list(csv.DictReader(stream))
    with (HERE / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    counts = Counter(row["split"] for row in mapping)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected split counts: {dict(counts)}")

    contact_sheets(mapping)
    split_coverage(manifest)
    frame = acquisition_conditions(manifest)
    frame.to_csv(FIGURES / "acquisition_conditions.csv", index=False)
    condition_plot(
        frame,
        ["water level (m)", "Hs (m)", "mean direction (°)", "peak frequency (Hz)"],
        (1, 4),
        "split_conditions.png",
        "Combined wave-condition coverage (one point per acquisition)",
    )
    condition_plot(
        frame,
        [
            "local hour (EDT)", "solar elevation (°)", "solar azimuth (°)",
            "wind speed (m/s)", "wind direction (°)", "wind gust (m/s)",
            "total cloud cover", "low cloud cover", "mid cloud cover", "high cloud cover",
        ],
        (2, 5),
        "lighting_met_conditions.png",
        "Combined lighting and meteorological coverage (one point per acquisition)",
    )
    print("wrote six split figures and acquisition_conditions.csv")


if __name__ == "__main__":
    main()
