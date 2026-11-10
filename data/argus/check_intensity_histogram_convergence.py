#!/usr/bin/env python3
"""Check nonzero-intensity histogram convergence versus all active frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropout_common import (
    DEFAULT_DATASET,
    DEFAULT_VIDEO_DIRS,
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
DEFAULT_SIZES = (25, 50, 100, 200, 500, 1000)
DEFAULT_CSV = HERE / "nonzero_pixel_intensity_convergence.csv"
DEFAULT_FIGURE = HERE / "nonzero_pixel_intensity_convergence.png"


@dataclass(frozen=True)
class VideoHistograms:
    video: str
    sample_sizes: tuple[int, ...]
    sampled_frames: np.ndarray
    sampled_counts: np.ndarray
    exact_frames: int
    exact_counts: np.ndarray


def analyze_video(path: Path, sample_sizes: tuple[int, ...]) -> VideoHistograms:
    scan = scan_video(path)
    eligible = np.flatnonzero(scan.active_nonempty)
    if eligible.size == 0:
        raise ValueError(f"no nonempty active frames in {path}")

    selected = np.zeros((len(sample_sizes), scan.active_frames), dtype=bool)
    frames_used = np.zeros(len(sample_sizes), dtype=np.int32)
    for index, requested in enumerate(sample_sizes):
        chosen = evenly_spaced(eligible, requested)
        selected[index, chosen] = True
        frames_used[index] = chosen.size

    sampled_counts = np.zeros((len(sample_sizes), 256), dtype=np.int64)
    exact_counts = np.zeros(256, dtype=np.int64)
    exact_frames = 0
    for frame_index, frame in active_gray_frames(path, scan):
        values = frame[scan.valid_mask & (frame != 0)]
        if values.size == 0:
            continue
        histogram = np.bincount(values, minlength=256)
        exact_counts += histogram
        exact_frames += 1
        for sample_index in np.flatnonzero(selected[:, frame_index]):
            sampled_counts[sample_index] += histogram

    if exact_frames != eligible.size:
        raise RuntimeError(
            f"measured {exact_frames}/{eligible.size} nonempty frames from {path}"
        )
    return VideoHistograms(
        video=path.name,
        sample_sizes=sample_sizes,
        sampled_frames=frames_used,
        sampled_counts=sampled_counts,
        exact_frames=exact_frames,
        exact_counts=exact_counts,
    )


def convergence_rows(results: list[VideoHistograms]) -> list[dict[str, float | int]]:
    sizes = results[0].sample_sizes
    sampled_counts = np.stack([item.sampled_counts for item in results], axis=1)
    exact_counts = np.stack([item.exact_counts for item in results])
    exact_video_pdf = normalize_histograms(exact_counts)
    intensity = np.arange(256)

    rows: list[dict[str, float | int]] = []
    for sample_index, size in enumerate(sizes):
        counts = sampled_counts[sample_index]
        video_pdf = normalize_histograms(counts)
        video_delta = video_pdf - exact_video_pdf
        video_tv = 0.5 * np.abs(video_delta).sum(axis=1)
        video_ks = np.abs(np.cumsum(video_delta, axis=1)).max(axis=1)

        pooled_pdf = normalize_histograms(counts.sum(axis=0)[None])[0]
        exact_pooled_pdf = normalize_histograms(exact_counts.sum(axis=0)[None])[0]
        delta = pooled_pdf - exact_pooled_pdf
        cdf_delta = np.cumsum(delta)
        sampled_mean = float(pooled_pdf @ intensity)
        exact_mean = float(exact_pooled_pdf @ intensity)
        rows.append(
            {
                "sample_frames_per_video": size,
                "actual_sampled_frames": int(
                    sum(item.sampled_frames[sample_index] for item in results)
                ),
                "pooled_total_variation": float(0.5 * np.abs(delta).sum()),
                "pooled_max_cdf_error": float(np.abs(cdf_delta).max()),
                "pooled_wasserstein_intensity_levels": float(np.abs(cdf_delta).sum()),
                "pooled_mean_intensity_error": sampled_mean - exact_mean,
                "pooled_median_intensity_error": (
                    histogram_quantile(pooled_pdf, 0.5)
                    - histogram_quantile(exact_pooled_pdf, 0.5)
                ),
                "pooled_p05_intensity_error": (
                    histogram_quantile(pooled_pdf, 0.05)
                    - histogram_quantile(exact_pooled_pdf, 0.05)
                ),
                "pooled_p95_intensity_error": (
                    histogram_quantile(pooled_pdf, 0.95)
                    - histogram_quantile(exact_pooled_pdf, 0.95)
                ),
                "mean_per_video_total_variation": float(video_tv.mean()),
                "p95_per_video_total_variation": float(np.quantile(video_tv, 0.95)),
                "mean_per_video_max_cdf_error": float(video_ks.mean()),
                "p95_per_video_max_cdf_error": float(np.quantile(video_ks, 0.95)),
            }
        )
    return rows


def write_figure(path: Path, rows: list[dict[str, float | int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = np.asarray([row["sample_frames_per_video"] for row in rows])
    mean_tv = np.asarray([row["mean_per_video_total_variation"] for row in rows])
    p95_tv = np.asarray([row["p95_per_video_total_variation"] for row in rows])
    mean_ks = np.asarray([row["mean_per_video_max_cdf_error"] for row in rows])
    p95_ks = np.asarray(
        [row["p95_per_video_max_cdf_error"] for row in rows]
    )

    fig, (axis_tv, axis_cdf) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axis_tv.plot(size, mean_tv, "o-", label="mean across videos")
    axis_tv.plot(size, p95_tv, "s-", label="95th percentile across videos")
    axis_tv.set(
        xscale="log",
        yscale="log",
        xlabel="sampled nonempty frames per video",
        ylabel="total-variation distance",
        title="PDF convergence",
    )
    axis_tv.grid(alpha=0.25, which="both")
    axis_tv.set_xticks(size, labels=[str(value) for value in size])
    axis_tv.legend(frameon=False, fontsize=8)

    axis_cdf.plot(size, mean_ks, "o-", label="mean across videos")
    axis_cdf.plot(size, p95_ks, "s-", label="95th percentile across videos")
    axis_cdf.set(
        xscale="log",
        yscale="log",
        xlabel="sampled nonempty frames per video",
        ylabel="maximum absolute CDF error",
        title="CDF convergence",
    )
    axis_cdf.grid(alpha=0.25, which="both")
    axis_cdf.set_xticks(size, labels=[str(value) for value in size])
    axis_cdf.legend(frameon=False, fontsize=8)
    fig.suptitle("Nonzero pixel-intensity histogram convergence")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_rows(rows: list[dict[str, float | int]]) -> None:
    print("\nframes  mean-video TV  p95-video TV  mean-video CDF  p95-video CDF")
    for row in rows:
        print(
            f"{row['sample_frames_per_video']:>6}  "
            f"{row['mean_per_video_total_variation']:>13.5f}  "
            f"{row['p95_per_video_total_variation']:>12.5f}  "
            f"{row['mean_per_video_max_cdf_error']:>14.5f}  "
            f"{row['p95_per_video_max_cdf_error']:>13.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--video-dir", type=Path, action="append", dest="video_dirs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    sample_sizes = tuple(sorted(set(args.sample_sizes)))
    if not sample_sizes or sample_sizes[0] < 1:
        raise SystemExit("sample sizes must be positive")

    video_dirs = args.video_dirs or list(DEFAULT_VIDEO_DIRS)
    paths, _ = resolve_dataset_videos(args.dataset, video_dirs)
    print(
        f"Analyzing {len(paths)} videos at sample sizes {sample_sizes}; "
        f"exact reference uses every nonempty active frame"
    )
    results = parallel_map(paths, analyze_video, args.workers, sample_sizes)
    rows = convergence_rows(results)
    write_csv(args.output, rows)
    write_figure(args.figure, rows)
    print_rows(rows)
    print(f"\nConvergence CSV: {args.output}")
    print(f"Convergence figure: {args.figure}")


if __name__ == "__main__":
    main()
