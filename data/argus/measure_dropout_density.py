#!/usr/bin/env python3
"""Bin dropout density and sample nonzero grayscale intensity histograms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.datetime_utils import (
    get_time_resolution_for_date,
    parse_datetime_from_filename,
)
from dropout_common import (
    DEFAULT_DATASET,
    DEFAULT_METADATA_DIR,
    DEFAULT_VIDEO_DIRS,
    active_capture_array,
    active_gray_frames,
    evenly_spaced,
    histogram_quantile,
    normalize_histograms,
    parallel_map,
    resolve_dataset_videos,
    scan_video,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DENSITY_CSV = HERE / "dropout_density_distribution.csv"
DEFAULT_INTENSITY_NPZ = HERE / "nonzero_pixel_intensity_histograms.npz"
DEFAULT_INTENSITY_CSV = HERE / "nonzero_pixel_intensity_histogram.csv"
DEFAULT_INTENSITY_FIGURE = HERE / "nonzero_pixel_intensity_histograms.png"
DENSITY_BINS = (
    ("clean", 0.0, 0.0),
    (">0-20%", 0.0, 0.2),
    (">20-40%", 0.2, 0.4),
    (">40-60%", 0.4, 0.6),
    (">60-80%", 0.6, 0.8),
    (">80-<100%", 0.8, 1.0),
    ("100%", 1.0, 1.0),
)


@dataclass(frozen=True)
class DensitySeries:
    name: str
    fps: float
    spatial_units: int
    density: np.ndarray
    histogram: np.ndarray | None = None
    histogram_frames: int = 0


def analyze_video(
    path: Path, rates: dict[Path, float], histogram_frames: int
) -> DensitySeries:
    scan = scan_video(path)
    fps = rates.get(path) or get_time_resolution_for_date(
        parse_datetime_from_filename(path.name)
    )[1]
    eligible = np.flatnonzero(scan.active_nonempty)
    selected = np.zeros(scan.active_frames, dtype=bool)
    selected[evenly_spaced(eligible, histogram_frames)] = True

    density = np.empty(scan.active_frames, dtype=float)
    histogram = np.zeros(256, dtype=np.int64)
    for frame_index, frame in active_gray_frames(path, scan):
        missing = scan.valid_mask & (frame == 0)
        density[frame_index] = missing.sum() / scan.valid_pixels
        if selected[frame_index]:
            histogram += np.bincount(
                frame[scan.valid_mask & (frame != 0)], minlength=256
            )
    return DensitySeries(
        path.name,
        fps,
        scan.valid_pixels,
        density,
        histogram,
        int(selected.sum()),
    )


def metadata_series(paths: list[Path]) -> list[DensitySeries]:
    results = []
    for path in paths:
        stamp, fps, _, capture = active_capture_array(path)
        results.append(
            DensitySeries(stamp, fps, capture.shape[1], (~capture).mean(axis=1))
        )
    return results


def bin_mask(values: np.ndarray, label: str, lower: float, upper: float) -> np.ndarray:
    if label == "clean":
        return values == 0
    if label == "100%":
        return values == 1
    if label == ">80-<100%":
        return (values > lower) & (values < upper)
    return (values > lower) & (values <= upper)


def summarize_density(scope: str, series: list[DensitySeries]) -> list[dict[str, object]]:
    total_time = sum(item.density.size / item.fps for item in series)
    total_capacity = sum(
        item.density.size * item.spatial_units / item.fps for item in series
    )
    total_missing = sum(
        item.density.sum() * item.spatial_units / item.fps for item in series
    )
    rows = []
    for label, lower, upper in DENSITY_BINS:
        frames = duration = capacity = missing = 0.0
        for item in series:
            mask = bin_mask(item.density, label, lower, upper)
            count = int(mask.sum())
            frames += count
            duration += count / item.fps
            capacity += count * item.spatial_units / item.fps
            missing += item.density[mask].sum() * item.spatial_units / item.fps
        rows.append(
            {
                "scope": scope,
                "density_band": label,
                "lower_fraction": lower,
                "upper_fraction": upper,
                "frame_count": int(frames),
                "duration_seconds": duration,
                "time_fraction": duration / total_time,
                "valid_spacetime_fraction": capacity / total_capacity,
                "missing_spacetime_contribution": missing / total_capacity,
                "share_of_all_missing": missing / total_missing if total_missing else 0.0,
            }
        )
    return rows


def print_density(rows: list[dict[str, object]]) -> None:
    print(f"\n{rows[0]['scope']}")
    print("  density       time      missing capacity   share of missing")
    for row in rows:
        print(
            f"  {row['density_band']:<11} {row['time_fraction']:>9.3%} "
            f"{row['missing_spacetime_contribution']:>17.3%} "
            f"{row['share_of_all_missing']:>18.3%}"
        )


def write_histograms(
    series: list[DensitySeries], npz_path: Path, csv_path: Path
) -> dict[str, float | int]:
    if any(item.histogram is None for item in series):
        raise ValueError("histograms require decoded video series")
    counts = np.stack([item.histogram for item in series])
    aggregate = counts.sum(axis=0, dtype=np.int64)
    aggregate[0] = 0
    pdf = normalize_histograms(aggregate[None])[0]
    cdf = np.cumsum(pdf)
    intensity = np.arange(256)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        video=np.asarray([item.name for item in series]),
        intensity=intensity,
        counts=counts,
        aggregate_counts=aggregate,
        frames_used=np.asarray([item.histogram_frames for item in series]),
    )
    rows = [
        {
            "intensity": int(value),
            "count": int(aggregate[value]),
            "probability": float(pdf[value]),
            "cumulative": float(cdf[value]),
        }
        for value in intensity
    ]
    write_csv(csv_path, rows)

    total = int(aggregate.sum())
    mean = float(pdf @ intensity)
    return {
        "videos": len(series),
        "frames": sum(item.histogram_frames for item in series),
        "pixels": total,
        "mean": mean,
        "standard_deviation": float(np.sqrt(pdf @ (intensity - mean) ** 2)),
        "p05": histogram_quantile(pdf, 0.05),
        "p25": histogram_quantile(pdf, 0.25),
        "median": histogram_quantile(pdf, 0.50),
        "p75": histogram_quantile(pdf, 0.75),
        "p95": histogram_quantile(pdf, 0.95),
    }


def print_histogram_summary(summary: dict[str, float | int]) -> None:
    print("\nNonzero grayscale intensity")
    print(f"  videos / sampled frames: {summary['videos']:,} / {summary['frames']:,}")
    print(f"  nonzero pixels: {summary['pixels']:,}")
    print(
        f"  mean +/- standard deviation: {summary['mean']:.3f} +/- "
        f"{summary['standard_deviation']:.3f}"
    )
    print(
        "  p05 / p25 / median / p75 / p95: "
        f"{summary['p05']} / {summary['p25']} / {summary['median']} / "
        f"{summary['p75']} / {summary['p95']}"
    )


def write_histogram_figure(series: list[DensitySeries], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    counts = np.stack([item.histogram for item in series])
    pdf = normalize_histograms(counts)
    aggregate_pdf = normalize_histograms(counts.sum(axis=0)[None])[0]
    intensity = np.arange(256)
    means = pdf @ intensity
    medians = np.asarray([histogram_quantile(row, 0.5) for row in pdf])
    order = np.argsort(medians)
    cmap = plt.get_cmap("cividis")
    norm = Normalize(float(means.min()), float(means.max()))

    fig, (axis_pdf, axis_map) = plt.subplots(1, 2, figsize=(13, 5.2))
    for row, mean in zip(pdf, means, strict=True):
        axis_pdf.plot(intensity[1:], row[1:], color=cmap(norm(mean)), alpha=0.55, lw=0.8)
    axis_pdf.plot(intensity[1:], aggregate_pdf[1:], color="black", lw=2.2, label="pooled")
    for probability, style in ((0.05, ":"), (0.50, "--"), (0.95, ":")):
        axis_pdf.axvline(
            histogram_quantile(aggregate_pdf, probability), color="black", ls=style, alpha=0.55
        )
    axis_pdf.set(
        xlim=(1, 255),
        xlabel="grayscale intensity (zero excluded)",
        ylabel="probability per intensity level",
        title="Per-video and pooled histograms",
    )
    axis_pdf.grid(alpha=0.2)
    axis_pdf.legend(frameon=False)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axis_pdf, label="per-video mean", pad=0.02
    )

    image = axis_map.imshow(
        pdf[order, 1:],
        aspect="auto",
        interpolation="nearest",
        extent=(1, 255, len(series), 0),
        cmap="magma",
        vmin=0,
        vmax=float(np.quantile(pdf[:, 1:], 0.995)),
    )
    tick_rows = np.unique(np.linspace(0, len(series) - 1, 8, dtype=int))
    labels = [series[order[index]].name.replace("ArgusFF_", "")[:15] for index in tick_rows]
    axis_map.set_yticks(tick_rows + 0.5, labels=labels)
    axis_map.set(
        xlim=(1, 255),
        xlabel="grayscale intensity (zero excluded)",
        ylabel="video (sorted by median)",
        title="Normalized histogram by video",
    )
    fig.colorbar(image, ax=axis_map, label="probability per intensity level", pad=0.02)
    fig.suptitle(
        f"Argus nonzero pixel intensity — {len(series)} videos, "
        f"{sum(item.histogram_frames for item in series):,} sampled frames"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--video-dir", type=Path, action="append", dest="video_dirs")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--histogram-frames", type=int, default=500)
    parser.add_argument("--output", type=Path, default=DEFAULT_DENSITY_CSV)
    parser.add_argument("--intensity-npz", type=Path, default=DEFAULT_INTENSITY_NPZ)
    parser.add_argument("--intensity-csv", type=Path, default=DEFAULT_INTENSITY_CSV)
    parser.add_argument("--intensity-figure", type=Path, default=DEFAULT_INTENSITY_FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.histogram_frames < 1:
        raise SystemExit("workers and histogram frames must be positive")
    metadata_paths = sorted(args.metadata_dir.glob("*_headerData.pickle"))
    all_metadata = metadata_series(metadata_paths)
    rows: list[dict[str, object]] = []

    if not args.metadata_only:
        paths, rates = resolve_dataset_videos(
            args.dataset, args.video_dirs or list(DEFAULT_VIDEO_DIRS)
        )
        print(f"Decoding {len(paths)} videos with {args.workers} worker(s)")
        pixels = parallel_map(
            paths, analyze_video, args.workers, rates, args.histogram_frames
        )
        histogram_summary = write_histograms(
            pixels, args.intensity_npz, args.intensity_csv
        )
        write_histogram_figure(pixels, args.intensity_figure)
        print_histogram_summary(histogram_summary)

        pixel_rows = summarize_density("analysis_pixels", pixels)
        print_density(pixel_rows)
        rows.extend(pixel_rows)
        stamps = {
            parse_datetime_from_filename(path.name).strftime("%Y%m%dT%H%M%SZ")
            for path in paths
        }
        selected_rows = summarize_density(
            "analysis_metadata", [item for item in all_metadata if item.name in stamps]
        )
        print_density(selected_rows)
        rows.extend(selected_rows)

    metadata_rows = summarize_density("full_metadata", all_metadata)
    print_density(metadata_rows)
    rows.extend(metadata_rows)
    write_csv(args.output, rows)
    print(f"\nDensity distribution: {args.output}")


if __name__ == "__main__":
    main()
