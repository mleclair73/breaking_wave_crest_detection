#!/usr/bin/env python3
"""Measure intermittent zero-valued pixels in Argus analysis videos.

Always-zero footprint pixels and all-zero endpoint padding are excluded.
Missing samples separated by at most ``--timeout-seconds`` form one episode;
the episode metric counts its complete duration and union spatial footprint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common.datetime_utils import (
    get_time_resolution_for_date,
    parse_datetime_from_filename,
)
from dropout_common import (
    DEFAULT_DATASET,
    DEFAULT_VIDEO_DIRS,
    VideoScan,
    active_gray_frames,
    frequency_stats,
    parallel_map,
    resolve_dataset_videos,
    scan_video,
    write_csv,
)


DEFAULT_OUTPUT = Path(__file__).with_name("flickering_zero_fractions.csv")


@dataclass(frozen=True)
class VideoResult:
    video: str
    path: str
    width: int
    height: int
    fps: float
    decoded_frames: int
    active_frames: int
    leading_all_zero_frames: int
    internal_all_zero_frames: int
    trailing_all_zero_frames: int
    valid_pixels: int
    flickering_pixels: int
    missing_pixel_frames: int
    missing_frames: int
    missing_frame_fraction: float
    episode_count: int
    episode_frames: int
    episode_time_fraction: float
    episode_pixel_frames: int
    episode_spacetime_fraction: float
    dropout_frequency_resolution_hz: float
    dropout_dominant_frequency_hz: float
    dropout_power_p05_hz: float
    dropout_power_median_hz: float
    dropout_power_p95_hz: float
    spatial_affected_fraction: float
    temporal_missing_fraction_at_affected_pixels: float
    spacetime_missing_fraction: float


def episode_metrics(
    path: Path, scan: VideoScan, timeout_frames: int
) -> tuple[dict[str, int], np.ndarray]:
    """Measure joined dropout episodes and per-frame missing density."""
    density = np.empty(scan.active_frames, dtype=float)
    missing_frames = episode_count = episode_frames = episode_pixel_frames = 0
    start: int | None = None
    last: int | None = None
    footprint: np.ndarray | None = None

    def finish() -> tuple[int, int]:
        assert start is not None and last is not None and footprint is not None
        duration = last - start + 1
        return duration, duration * int(footprint.sum())

    for frame_index, frame in active_gray_frames(path, scan):
        missing = scan.valid_mask & (frame == 0)
        density[frame_index] = missing.sum() / scan.valid_pixels
        if not missing.any():
            continue
        missing_frames += 1
        if last is None or frame_index - last > timeout_frames:
            if last is not None:
                duration, pixel_frames = finish()
                episode_frames += duration
                episode_pixel_frames += pixel_frames
            episode_count += 1
            start = frame_index
            footprint = missing.copy()
        else:
            assert footprint is not None
            footprint |= missing
        last = frame_index

    if last is not None:
        duration, pixel_frames = finish()
        episode_frames += duration
        episode_pixel_frames += pixel_frames
    return {
        "missing_frames": missing_frames,
        "episode_count": episode_count,
        "episode_frames": episode_frames,
        "episode_pixel_frames": episode_pixel_frames,
    }, density


def analyze_video(
    path: Path, timeout_seconds: float, rates: dict[Path, float]
) -> VideoResult:
    scan = scan_video(path)
    fps = rates.get(path) or get_time_resolution_for_date(
        parse_datetime_from_filename(path.name)
    )[1]
    episodes, density = episode_metrics(
        path, scan, max(1, round(timeout_seconds * fps))
    )

    valid_pixels = scan.valid_pixels
    flickering_pixels = int(scan.flickering_mask.sum())
    missing_pixel_frames = int(scan.zero_counts[scan.valid_mask].sum(dtype=np.uint64))
    valid_pixel_frames = scan.active_frames * valid_pixels
    affected_pixel_frames = scan.active_frames * flickering_pixels
    spectrum = frequency_stats(density, fps)

    return VideoResult(
        video=path.name,
        path=str(path),
        width=scan.width,
        height=scan.height,
        fps=fps,
        decoded_frames=scan.decoded_frames,
        active_frames=scan.active_frames,
        leading_all_zero_frames=scan.leading_blank,
        internal_all_zero_frames=scan.internal_blank,
        trailing_all_zero_frames=scan.trailing_blank,
        valid_pixels=valid_pixels,
        flickering_pixels=flickering_pixels,
        missing_pixel_frames=missing_pixel_frames,
        missing_frames=episodes["missing_frames"],
        missing_frame_fraction=episodes["missing_frames"] / scan.active_frames,
        episode_count=episodes["episode_count"],
        episode_frames=episodes["episode_frames"],
        episode_time_fraction=episodes["episode_frames"] / scan.active_frames,
        episode_pixel_frames=episodes["episode_pixel_frames"],
        episode_spacetime_fraction=episodes["episode_pixel_frames"] / valid_pixel_frames,
        dropout_frequency_resolution_hz=spectrum["frequency_resolution_hz"],
        dropout_dominant_frequency_hz=spectrum["dominant_frequency_hz"],
        dropout_power_p05_hz=spectrum["power_p05_hz"],
        dropout_power_median_hz=spectrum["power_median_hz"],
        dropout_power_p95_hz=spectrum["power_p95_hz"],
        spatial_affected_fraction=flickering_pixels / valid_pixels,
        temporal_missing_fraction_at_affected_pixels=(
            missing_pixel_frames / affected_pixel_frames
            if affected_pixel_frames
            else 0.0
        ),
        spacetime_missing_fraction=missing_pixel_frames / valid_pixel_frames,
    )


def aggregate(results: list[VideoResult]) -> dict[str, float | int]:
    active_seconds = sum(item.active_frames / item.fps for item in results)
    valid_pixel_seconds = sum(
        item.active_frames * item.valid_pixels / item.fps for item in results
    )
    affected_pixel_seconds = sum(
        item.active_frames * item.flickering_pixels / item.fps for item in results
    )
    missing_pixel_seconds = sum(item.missing_pixel_frames / item.fps for item in results)
    episode_pixel_seconds = sum(item.episode_pixel_frames / item.fps for item in results)
    frequencies = np.asarray(
        [item.dropout_dominant_frequency_hz for item in results], dtype=float
    )
    finite = frequencies[np.isfinite(frequencies)]
    affected = sum(item.flickering_pixels > 0 for item in results)
    return {
        "videos": len(results),
        "affected_videos": affected,
        "affected_video_fraction": affected / len(results),
        "active_seconds": active_seconds,
        "missing_time_fraction": sum(
            item.missing_frames / item.fps for item in results
        )
        / active_seconds,
        "episode_time_fraction": sum(
            item.episode_frames / item.fps for item in results
        )
        / active_seconds,
        "episode_spacetime_fraction": episode_pixel_seconds / valid_pixel_seconds,
        "spacetime_missing_fraction": missing_pixel_seconds / valid_pixel_seconds,
        "spatial_affected_fraction": affected_pixel_seconds / valid_pixel_seconds,
        "dominant_frequency_min_hz": float(finite.min()) if finite.size else np.nan,
        "dominant_frequency_median_hz": (
            float(np.median(finite)) if finite.size else np.nan
        ),
        "dominant_frequency_max_hz": float(finite.max()) if finite.size else np.nan,
    }


def print_summary(summary: dict[str, float | int]) -> None:
    print("\nAggregate (pooled by acquisition time)")
    print(f"  videos: {summary['videos']:,}")
    print(
        f"  videos with flickering zeros: {summary['affected_videos']:,} "
        f"({summary['affected_video_fraction']:.6%})"
    )
    print(f"  active video time: {summary['active_seconds'] / 3600:.3f} h")
    print(f"  time containing missing pixels: {summary['missing_time_fraction']:.6%}")
    print(f"  time in joined episodes: {summary['episode_time_fraction']:.6%}")
    print(
        "  episode duration/footprint space-time: "
        f"{summary['episode_spacetime_fraction']:.6%}"
    )
    print(f"  actual missing space-time: {summary['spacetime_missing_fraction']:.6%}")
    print(
        "  dominant frequencies (min / median / max): "
        f"{summary['dominant_frequency_min_hz']:.6g} / "
        f"{summary['dominant_frequency_median_hz']:.6g} / "
        f"{summary['dominant_frequency_max_hz']:.6g} Hz"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--video-dir", type=Path, action="append", dest="video_dirs")
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.timeout_seconds < 0:
        raise SystemExit("workers must be positive and timeout nonnegative")
    video_dirs = args.video_dirs or list(DEFAULT_VIDEO_DIRS)
    if args.all_videos:
        paths = sorted(path for directory in video_dirs for path in directory.glob("*.avi"))
        rates: dict[Path, float] = {}
    else:
        paths, rates = resolve_dataset_videos(args.dataset, video_dirs)
    if not paths:
        raise FileNotFoundError("no AVI videos found")

    print(
        f"Analyzing {len(paths)} videos with {args.workers} worker(s); "
        f"episode timeout={args.timeout_seconds:g} s"
    )
    results = parallel_map(paths, analyze_video, args.workers, args.timeout_seconds, rates)
    write_csv(args.output, [asdict(item) for item in results])
    print_summary(aggregate(results))
    print(f"\nPer-video results: {args.output}")


if __name__ == "__main__":
    main()
