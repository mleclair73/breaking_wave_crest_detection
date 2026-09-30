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
OOD_CANDIDATES = FIGURES / "ood_candidates.csv"
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


def ood_candidate_plot(frame: pd.DataFrame) -> None:
    candidates = pd.read_csv(OOD_CANDIDATES)
    if set(candidates["priority"]) != {"primary", "edge", "alternate"}:
        raise ValueError("OOD candidates must contain primary, edge, and alternate rows")
    planned = candidates[candidates["target_crops"] > 0]
    if int(planned["target_crops"].sum()) != 20:
        raise ValueError("the provisional OOD sampling plan must contain exactly 20 crops")
    standard = planned[planned["priority"] != "alternate"]
    if not ((standard["camera_count"] == 6) & (standard["longest_complete_run_frames"] >= 480)).all():
        raise ValueError("every standard planned OOD source must have six cameras and >=480 complete frames")
    camera_limited = planned[planned["priority"] == "alternate"]
    if len(camera_limited) != 1 or int(camera_limited["target_crops"].sum()) != 1:
        raise ValueError("the plan must contain exactly one spatially audited camera-limited Tp example")
    existing_dates = set(pd.to_datetime(frame["acquisition_utc"], utc=True).dt.date)
    candidate_dates = set(pd.to_datetime(candidates["acquisition_utc"], utc=True).dt.date)
    if existing_dates & candidate_dates:
        raise ValueError("an OOD candidate date overlaps the combined train/validation/test dataset")
    required_edges = {
        "Hs_m": (0.35, 1.55),
        "peak_frequency_hz": (0.085, 0.21),
        "mean_direction_deg": (57.0, 90.0),
        "solar_elevation_deg": (1.0, 52.0),
        "solar_azimuth_deg": (95.0, 260.0),
        "water_level_m": (-0.40, 0.90),
    }
    for column, (low_ceiling, high_floor) in required_edges.items():
        if planned[column].min() > low_ceiling or planned[column].max() < high_floor:
            raise ValueError(f"planned OOD sources no longer bracket the required {column} low/high edges")
    acquired = pd.to_datetime(candidates["acquisition_utc"], utc=True)
    candidates["local hour (EDT)"] = (
        acquired.dt.hour + acquired.dt.minute / 60 + acquired.dt.second / 3600 - 4
    ) % 24
    candidates = candidates.rename(columns={
        "Hs_m": "Hs (m)",
        "peak_frequency_hz": "peak frequency (Hz)",
        "mean_direction_deg": "mean direction (°)",
        "solar_elevation_deg": "solar elevation (°)",
        "solar_azimuth_deg": "solar azimuth (°)",
        "water_level_m": "water level (m)",
    })
    panels = [
        ("Hs (m)", "mean direction (°)"),
        ("Hs (m)", "peak frequency (Hz)"),
        ("peak frequency (Hz)", "mean direction (°)"),
        ("solar azimuth (°)", "solar elevation (°)"),
        ("water level (m)", "Hs (m)"),
        ("water level (m)", "solar elevation (°)"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    candidate_styles = {
        "primary": {"color": "#d62728", "marker": "*", "size": 150, "prefix": "P"},
        "edge": {"color": "#17becf", "marker": "X", "size": 70, "prefix": "E"},
        "alternate": {"color": "#9467bd", "marker": "D", "size": 65, "prefix": "A"},
    }

    def convex_hull(points: np.ndarray) -> np.ndarray:
        """Return the 2-D monotonic-chain hull, closed for plotting."""
        points = np.unique(points, axis=0)
        if len(points) < 3:
            return points
        points = points[np.lexsort((points[:, 1], points[:, 0]))]

        def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
            first, second = first - origin, second - origin
            return float(first[0] * second[1] - first[1] * second[0])

        lower: list[np.ndarray] = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper: list[np.ndarray] = []
        for point in points[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        hull = np.asarray(lower[:-1] + upper[:-1])
        return np.vstack((hull, hull[0]))

    for panel_index, (axis, (x_column, y_column)) in enumerate(zip(axes.flat, panels)):
        complete = frame[[x_column, y_column]].dropna().to_numpy(float)
        hull = convex_hull(complete)
        if len(hull) >= 3:
            axis.fill(hull[:, 0], hull[:, 1], color="#777777", alpha=0.08, zorder=0)
            axis.plot(hull[:, 0], hull[:, 1], color="#555555", linestyle="--", linewidth=1, alpha=0.7)
        for split in SPLITS:
            selected = frame.loc[frame.split == split]
            axis.scatter(
                selected[x_column], selected[y_column], color=COLORS[split], s=42,
                alpha=0.85, zorder=2, label=split if panel_index == 0 else None,
            )
        for priority, style in candidate_styles.items():
            selected = candidates[candidates["priority"] == priority].sort_values("rank")
            for position, (_, row) in enumerate(selected.iterrows()):
                is_planned = int(row["target_crops"]) > 0
                axis.scatter(
                    row[x_column], row[y_column], marker=style["marker"], s=style["size"],
                    facecolor=style["color"] if is_planned else "none",
                    edgecolor="white" if is_planned else style["color"], linewidth=1.0,
                    zorder=4,
                    label=(
                        f"OOD {priority}" + ("" if is_planned else " (rejected/reserve)")
                        if panel_index == 0 and position == 0 else None
                    ),
                )
                axis.annotate(
                    f"{style['prefix']}{int(row['rank'])}",
                    (row[x_column], row[y_column]),
                    xytext=(5, 6 if position % 2 == 0 else -11), textcoords="offset points",
                    fontsize=7, color=style["color"], weight="bold",
                )
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Existing joint-condition cloud vs date-disjoint OOD candidates\n"
        "Dashed hulls: existing envelope; filled markers: reviewed plan; open markers: reserves",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURES / "ood_candidate_conditions.png", dpi=200, bbox_inches="tight")
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
    ood_candidate_plot(frame)
    print("wrote seven combined/OOD split figures and acquisition_conditions.csv")


if __name__ == "__main__":
    main()
