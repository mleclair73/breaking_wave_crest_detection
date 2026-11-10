#!/usr/bin/env python3
"""Plot the date-disjoint OOD source candidates against the full_split condition cloud.

Reads ood_candidates.csv (all screened candidates; source_candidates.csv is the
chosen subset) and ../full_split/figures/acquisition_conditions.csv, written by
full_split/build_split_figures.py. Writes ood_candidate_conditions.png here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OOD_CANDIDATES = HERE / "ood_candidates.csv"
FULL_SPLIT_CONDITIONS = HERE.parent / "full_split" / "figures" / "acquisition_conditions.csv"
SPLITS = ("train", "val", "test")
COLORS = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}


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
    figure.savefig(HERE / "ood_candidate_conditions.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    ood_candidate_plot(pd.read_csv(FULL_SPLIT_CONDITIONS))
    print("wrote ood_candidate_conditions.png")
