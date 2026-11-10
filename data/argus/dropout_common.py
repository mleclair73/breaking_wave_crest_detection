"""Shared I/O and numerical helpers for Argus dropout analyses."""

from __future__ import annotations

import csv
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import cv2
import numpy as np

from common.datetime_utils import (
    get_time_resolution_for_date,
    parse_datetime_from_filename,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "dunex_breaking_wave_crest_dataset.nc"
DEFAULT_VIDEO_DIRS = (REPO_ROOT / "data" / "video_dataset" / "argus",)
DEFAULT_METADATA_DIR = HERE / "metadata"

T = TypeVar("T")


@dataclass(frozen=True)
class VideoScan:
    """First-pass video geometry, active interval, and zero counts."""

    width: int
    height: int
    decoded_frames: int
    active_frames: int
    leading_blank: int
    internal_blank: int
    trailing_blank: int
    valid_mask: np.ndarray
    zero_counts: np.ndarray
    active_nonempty: np.ndarray

    @property
    def active_start(self) -> int:
        return self.leading_blank

    @property
    def active_end(self) -> int:
        return self.leading_blank + self.active_frames

    @property
    def valid_pixels(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def flickering_mask(self) -> np.ndarray:
        return self.valid_mask & (self.zero_counts > 0)


def gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        return frame[..., 0]
    raise ValueError(f"unexpected decoded frame shape: {frame.shape}")


def decoded_frames(capture: cv2.VideoCapture) -> Iterable[np.ndarray]:
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame


def inspect_frames(frames: Iterable[np.ndarray]) -> VideoScan:
    """Find the valid footprint and trim only all-zero endpoint padding."""
    shape: tuple[int, int] | None = None
    valid: np.ndarray | None = None
    zero_counts: np.ndarray | None = None
    decoded = active = leading = internal = pending = 0
    started = False
    has_data_flags: list[bool] = []

    for raw in frames:
        frame = gray(np.asarray(raw))
        if shape is None:
            shape = frame.shape
            valid = np.zeros(shape, dtype=bool)
            zero_counts = np.zeros(shape, dtype=np.uint32)
        elif frame.shape != shape:
            raise ValueError(f"frame geometry changed from {shape} to {frame.shape}")

        decoded += 1
        nonzero = frame != 0
        has_data = bool(nonzero.any())
        has_data_flags.append(has_data)
        if not started:
            if not has_data:
                leading += 1
                continue
            started = True
        if not has_data:
            pending += 1
            continue

        assert valid is not None and zero_counts is not None
        if pending:
            zero_counts += pending
            active += pending
            internal += pending
            pending = 0
        zero_counts += ~nonzero
        valid |= nonzero
        active += 1

    if shape is None:
        raise ValueError("video contains no decodable frames")
    if not started:
        raise ValueError("video contains only all-zero frames")
    assert valid is not None and zero_counts is not None
    return VideoScan(
        width=shape[1],
        height=shape[0],
        decoded_frames=decoded,
        active_frames=active,
        leading_blank=leading,
        internal_blank=internal,
        trailing_blank=pending,
        valid_mask=valid,
        zero_counts=zero_counts,
        active_nonempty=np.asarray(
            has_data_flags[leading : decoded - pending if pending else decoded],
            dtype=bool,
        ),
    )


def scan_video(path: Path) -> VideoScan:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    try:
        scan = inspect_frames(decoded_frames(capture))
    finally:
        capture.release()
    if reported > 0 and scan.decoded_frames != reported:
        raise RuntimeError(
            f"decoded {scan.decoded_frames}/{reported} reported frames from {path}"
        )
    return scan


def active_gray_frames(path: Path, scan: VideoScan) -> Iterable[tuple[int, np.ndarray]]:
    """Yield ``(active_index, gray_frame)`` over a scan's active interval."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not reopen video: {path}")
    capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    try:
        for frame_index, frame in enumerate(decoded_frames(capture)):
            if frame_index >= scan.active_end:
                break
            if frame_index >= scan.active_start:
                yield frame_index - scan.active_start, gray(frame)
    finally:
        capture.release()


def resolve_dataset_videos(
    dataset: Path, video_dirs: list[Path]
) -> tuple[list[Path], dict[Path, float]]:
    import xarray as xr

    by_name: dict[str, Path] = {}
    for directory in video_dirs:
        for path in directory.glob("*.avi"):
            if path.name in by_name:
                raise ValueError(f"duplicate video across input directories: {path.name}")
            by_name[path.name] = path

    with xr.open_dataset(dataset) as data:
        if "source" not in data:
            raise KeyError(f"{dataset} has no 'source' variable")
        sources = [
            Path(str(value).replace("_pred", "")).name for value in data.source.values
        ]
        frame_rates = (
            [float(value) for value in data.fps.values]
            if "fps" in data
            else [None] * len(sources)
        )
    if len(sources) != len(set(sources)):
        raise ValueError(f"duplicate video sources in {dataset}")
    specs = list(zip(sources, frame_rates, strict=True))
    missing = [name for name, _ in specs if name not in by_name]
    if missing:
        raise FileNotFoundError("unresolved dataset videos:\n  " + "\n  ".join(missing))
    paths = [by_name[name] for name, _ in specs]
    rates = {by_name[name]: rate for name, rate in specs if rate is not None}
    return paths, rates


def active_capture_array(path: Path) -> tuple[str, float, int, np.ndarray]:
    """Load a header capture array, apply cadence, and trim endpoint padding."""
    when = parse_datetime_from_filename(path.name)
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    with path.open("rb") as stream:
        metadata = pickle.load(stream)
    capture = np.asarray(metadata["captureArray"], dtype=bool)
    if capture.ndim != 2 or capture.shape[0] == 0:
        raise ValueError(f"invalid captureArray shape {capture.shape} in {path}")
    listed_cameras = capture.shape[1]
    _, fps = get_time_resolution_for_date(when)
    if fps == 1.0:
        capture = capture[::2]
    capture = capture[:, capture.any(axis=0)]
    if capture.shape[1] == 0:
        raise ValueError(f"no camera captured any frames in {path}")
    active = np.flatnonzero(capture.any(axis=1))
    if active.size == 0:
        raise ValueError(f"no active interval in {path}")
    return stamp, fps, listed_cameras, capture[active[0] : active[-1] + 1]


def frequency_stats(signal: np.ndarray, fps: float) -> dict[str, float]:
    """Dominant frequency and central 90% of non-DC spectral power."""
    resolution = fps / signal.size if signal.size else np.nan
    empty = {
        "frequency_resolution_hz": resolution,
        "dominant_frequency_hz": np.nan,
        "power_p05_hz": np.nan,
        "power_median_hz": np.nan,
        "power_p95_hz": np.nan,
    }
    if signal.size < 2:
        return empty
    centered = signal - signal.mean()
    if not np.any(centered):
        return empty
    frequencies = np.fft.rfftfreq(signal.size, d=1 / fps)[1:]
    power = np.abs(np.fft.rfft(centered * np.hanning(signal.size)))[1:] ** 2
    if not np.any(power):
        return empty
    cumulative = np.cumsum(power) / power.sum()

    def quantile(probability: float) -> float:
        index = min(int(np.searchsorted(cumulative, probability)), frequencies.size - 1)
        return float(frequencies[index])

    return {
        "frequency_resolution_hz": resolution,
        "dominant_frequency_hz": float(frequencies[np.argmax(power)]),
        "power_p05_hz": quantile(0.05),
        "power_median_hz": quantile(0.50),
        "power_p95_hz": quantile(0.95),
    }


def evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    """Select up to ``count`` evenly spaced entries from a sorted array."""
    positions = np.unique(
        np.linspace(0, values.size - 1, min(count, values.size), dtype=int)
    )
    return values[positions]


def normalize_histograms(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)


def histogram_quantile(pdf: np.ndarray, probability: float) -> int:
    return int(np.searchsorted(np.cumsum(pdf), probability))


def parallel_map(
    paths: list[Path], function: Callable[..., T], workers: int, *args: object
) -> list[T]:
    if workers == 1:
        results = []
        for index, path in enumerate(paths, 1):
            results.append(function(path, *args))
            print(f"[{index:>2}/{len(paths)}] {path.name}", flush=True)
        return results
    by_path: dict[Path, T] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, path, *args): path for path in paths}
        for index, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            by_path[path] = future.result()
            print(f"[{index:>2}/{len(paths)}] {path.name}", flush=True)
    return [by_path[path] for path in paths]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
