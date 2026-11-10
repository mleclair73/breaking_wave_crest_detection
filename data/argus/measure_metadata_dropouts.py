#!/usr/bin/env python3
"""Measure Argus dropout from camera capture metadata."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common.datetime_utils import parse_datetime_from_filename
from dropout_common import (
    DEFAULT_METADATA_DIR,
    active_capture_array,
    frequency_stats,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO_RESULTS = HERE / "flickering_zero_fractions.csv"
DEFAULT_OUTPUT = HERE / "metadata_dropout_fractions.csv"
DEFAULT_JOIN_OUTPUT = HERE / "metadata_video_dropout_join.csv"


@dataclass(frozen=True)
class MetadataResult:
    timestamp: str
    metadata_file: str
    fps: float
    listed_cameras: int
    active_cameras: int
    active_frames: int
    active_duration_seconds: float
    affected: bool
    missing_frames: int
    missing_time_fraction: float
    missing_camera_frames: int
    camera_time_missing_fraction: float
    episode_count: int
    episode_frames: int
    episode_time_fraction: float
    episode_camera_frames: int
    episode_camera_time_fraction: float
    dropout_frequency_resolution_hz: float
    dropout_dominant_frequency_hz: float
    dropout_power_p05_hz: float
    dropout_power_median_hz: float
    dropout_power_p95_hz: float


def episode_metrics(missing: np.ndarray, timeout_frames: int) -> tuple[int, int, int]:
    """Return episode count, duration frames, and camera-footprint frames."""
    count = frames = camera_frames = 0
    start: int | None = None
    last: int | None = None
    footprint: np.ndarray | None = None
    for index in np.flatnonzero(missing.any(axis=1)):
        index = int(index)
        if last is None or index - last > timeout_frames:
            if last is not None:
                assert start is not None and footprint is not None
                duration = last - start + 1
                frames += duration
                camera_frames += duration * int(footprint.sum())
            count += 1
            start = index
            footprint = missing[index].copy()
        else:
            assert footprint is not None
            footprint |= missing[index]
        last = index
    if last is not None:
        assert start is not None and footprint is not None
        duration = last - start + 1
        frames += duration
        camera_frames += duration * int(footprint.sum())
    return count, frames, camera_frames


def analyze_header(path: Path, timeout_seconds: float) -> MetadataResult:
    stamp, fps, listed_cameras, capture = active_capture_array(path)
    missing = ~capture
    active_frames, active_cameras = capture.shape
    episode_count, episode_frames, episode_camera_frames = episode_metrics(
        missing, max(1, round(timeout_seconds * fps))
    )
    spectrum = frequency_stats(missing.mean(axis=1), fps)
    return MetadataResult(
        timestamp=stamp,
        metadata_file=path.name,
        fps=fps,
        listed_cameras=listed_cameras,
        active_cameras=active_cameras,
        active_frames=active_frames,
        active_duration_seconds=active_frames / fps,
        affected=bool(missing.any()),
        missing_frames=int(missing.any(axis=1).sum()),
        missing_time_fraction=float(missing.any(axis=1).mean()),
        missing_camera_frames=int(missing.sum()),
        camera_time_missing_fraction=float(missing.mean()),
        episode_count=episode_count,
        episode_frames=episode_frames,
        episode_time_fraction=episode_frames / active_frames,
        episode_camera_frames=episode_camera_frames,
        episode_camera_time_fraction=(
            episode_camera_frames / (active_frames * active_cameras)
        ),
        dropout_frequency_resolution_hz=spectrum["frequency_resolution_hz"],
        dropout_dominant_frequency_hz=spectrum["dominant_frequency_hz"],
        dropout_power_p05_hz=spectrum["power_p05_hz"],
        dropout_power_median_hz=spectrum["power_median_hz"],
        dropout_power_p95_hz=spectrum["power_p95_hz"],
    )


def aggregate(results: list[MetadataResult]) -> dict[str, float | int]:
    active_seconds = sum(item.active_frames / item.fps for item in results)
    camera_seconds = sum(
        item.active_frames * item.active_cameras / item.fps for item in results
    )
    frequencies = np.asarray(
        [item.dropout_dominant_frequency_hz for item in results], dtype=float
    )
    finite = frequencies[np.isfinite(frequencies)]
    affected = sum(item.affected for item in results)
    return {
        "collections": len(results),
        "affected": affected,
        "active_hours": active_seconds / 3600,
        "missing_time_fraction": sum(
            item.missing_frames / item.fps for item in results
        )
        / active_seconds,
        "episode_time_fraction": sum(
            item.episode_frames / item.fps for item in results
        )
        / active_seconds,
        "camera_time_missing_fraction": sum(
            item.missing_camera_frames / item.fps for item in results
        )
        / camera_seconds,
        "episode_camera_time_fraction": sum(
            item.episode_camera_frames / item.fps for item in results
        )
        / camera_seconds,
        "episodes": sum(item.episode_count for item in results),
        "low_frequency": int((finite < 0.01).sum()),
        "mid_frequency": int(((finite >= 0.01) & (finite < 0.9)).sum()),
        "one_hz_frequency": int((finite >= 0.9).sum()),
    }


def print_summary(summary: dict[str, float | int]) -> None:
    print("\nFull header-metadata archive")
    print(f"  collections: {summary['collections']:,}")
    print(f"  active collection time: {summary['active_hours']:.3f} h")
    print(
        f"  collections affected: {summary['affected']:,} "
        f"({summary['affected'] / summary['collections']:.6%})"
    )
    print(f"  time containing camera dropout: {summary['missing_time_fraction']:.6%}")
    print(f"  time in joined episodes: {summary['episode_time_fraction']:.6%}")
    print(f"  camera-time actually missing: {summary['camera_time_missing_fraction']:.6%}")
    print(
        "  episode duration/camera footprint: "
        f"{summary['episode_camera_time_fraction']:.6%}"
    )
    print(f"  dropout episodes: {summary['episodes']:,}")
    print(
        "  dominant-frequency groups (<0.01 / 0.01--0.9 / >=0.9 Hz): "
        f"{summary['low_frequency']:,} / {summary['mid_frequency']:,} / "
        f"{summary['one_hz_frequency']:,}"
    )


def join_video_results(
    metadata: list[MetadataResult], video_path: Path
) -> list[dict[str, object]]:
    by_stamp = {item.timestamp: item for item in metadata}
    with video_path.open(newline="") as stream:
        videos = list(csv.DictReader(stream))
    fields = (
        ("pixel_missing_time_fraction", "missing_frame_fraction", "metadata_missing_time_fraction", "missing_time_fraction"),
        ("pixel_spacetime_missing_fraction", "spacetime_missing_fraction", "metadata_camera_time_missing_fraction", "camera_time_missing_fraction"),
        ("pixel_episode_time_fraction", "episode_time_fraction", "metadata_episode_time_fraction", "episode_time_fraction"),
        ("pixel_episode_spacetime_fraction", "episode_spacetime_fraction", "metadata_episode_camera_time_fraction", "episode_camera_time_fraction"),
    )
    rows = []
    for video in videos:
        stamp = parse_datetime_from_filename(video["video"]).strftime("%Y%m%dT%H%M%SZ")
        if stamp not in by_stamp:
            raise KeyError(f"no header metadata for {video['video']}")
        item = by_stamp[stamp]
        row: dict[str, object] = {"timestamp": item.timestamp, "video": video["video"]}
        for pixel_output, pixel_source, metadata_output, metadata_source in fields:
            row[pixel_output] = video[pixel_source]
            row[metadata_output] = getattr(item, metadata_source)
        rows.append(row)
    return rows


def print_validation(rows: list[dict[str, object]]) -> None:
    print(f"\nPixel/metadata validation ({len(rows)} matched videos)")
    pairs = (
        ("pixel_missing_time_fraction", "metadata_missing_time_fraction"),
        ("pixel_spacetime_missing_fraction", "metadata_camera_time_missing_fraction"),
        ("pixel_episode_time_fraction", "metadata_episode_time_fraction"),
        ("pixel_episode_spacetime_fraction", "metadata_episode_camera_time_fraction"),
    )
    for pixel_key, metadata_key in pairs:
        pixel = np.asarray([float(row[pixel_key]) for row in rows])
        metadata = np.asarray([float(row[metadata_key]) for row in rows])
        correlation = np.corrcoef(pixel, metadata)[0, 1]
        mae = np.mean(np.abs(pixel - metadata))
        print(f"  {pixel_key}: r={correlation:.6f}, MAE={mae:.6%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--video-results", type=Path, default=DEFAULT_VIDEO_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--join-output", type=Path, default=DEFAULT_JOIN_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_seconds < 0:
        raise SystemExit("--timeout-seconds cannot be negative")
    paths = sorted(args.metadata_dir.glob("*_headerData.pickle"))
    if not paths:
        raise FileNotFoundError(f"no header metadata under {args.metadata_dir}")
    results = [analyze_header(path, args.timeout_seconds) for path in paths]
    write_csv(args.output, [asdict(item) for item in results])
    print_summary(aggregate(results))
    print(f"  per-collection CSV: {args.output}")

    if args.video_results.is_file():
        joined = join_video_results(results, args.video_results)
        write_csv(args.join_output, joined)
        print_validation(joined)
        print(f"  joined validation CSV: {args.join_output}")


if __name__ == "__main__":
    main()
