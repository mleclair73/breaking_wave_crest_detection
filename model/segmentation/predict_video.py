#!/usr/bin/env python3
"""Production video inference for the selected breaking-crest model."""

from __future__ import annotations

import argparse
import gc
import queue
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import dunex_paths
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from checkpoint import clean_state_dict, seed_inference
from models import build_model
from tiling import predict_tiled, predict_tiled_many


PRODUCTION_MODEL_ID = "segnext_t_learned_up_skip_4_2"
PRODUCTION_THRESHOLD = 0.48


@dataclass(frozen=True)
class _ReaderFailure:
    error: Exception


def get_device(device_name: str | None = None) -> torch.device:
    """Resolve an explicit device or auto-detect CUDA, MPS, then CPU."""
    if device_name and device_name.lower() != "auto":
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_memory(device):
    """Clean up GPU/CPU memory."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def load_model(checkpoint_path, device="cpu"):
    """Reconstruct the production model and load every tensor strictly."""
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain config and model_state_dict")

    config = dict(checkpoint["config"])
    if "seed" not in config:
        raise ValueError("Production checkpoint config must contain seed")
    model_id = config.get("id")
    if model_id != PRODUCTION_MODEL_ID:
        raise ValueError(
            f"Production inference requires {PRODUCTION_MODEL_ID!r}; "
            f"checkpoint contains {model_id!r}"
        )
    if config.get("arch") != "segnext":
        raise ValueError(f"Unexpected production architecture: {config.get('arch')!r}")
    if config.get("decoder_type") != "learned_up":
        raise ValueError(f"Unexpected production decoder: {config.get('decoder_type')!r}")
    if list(config.get("decoder_skips", [])) != [4, 2]:
        raise ValueError("Production checkpoint must use H/4 and H/2 decoder skips")

    # The checkpoint contains the complete encoder state, so reconstruction
    # must never request pretrained weights or a network download.
    config["encoder_pretrained"] = False
    model = build_model(config)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model: {model_id}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Strictly loaded {len(state_dict)} checkpoint tensors")
    return model, config


class VideoFrameReader:
    """Threaded video frame reader with buffer for async I/O."""
    def __init__(self, video_path, buffer_size=50, max_frames=None):
        self.video_path = str(video_path)
        self.buffer_size = buffer_size
        self.max_frames = max_frames
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if self.total_frames <= 0:
            raise ValueError(f"Video reports no frames: {video_path}")
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self.expected_frames = (
            min(self.total_frames, max_frames)
            if max_frames is not None
            else self.total_frames
        )

        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reader_thread.start()

    def _enqueue(self, item):
        """Queue one frame or terminal message unless the reader is stopping."""
        while not self.stop_event.is_set():
            try:
                self.frame_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _read_frames(self):
        """Background thread that reads frames from disk."""
        cap = None
        terminal = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not reopen video: {self.video_path}")
            frame_count = 0
            while (
                frame_count < self.expected_frames
                and not self.stop_event.is_set()
            ):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Decoded {frame_count}/{self.expected_frames} expected "
                        f"frames from {self.video_path}"
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not self._enqueue(frame):
                    return
                frame_count += 1

            if (
                not self.stop_event.is_set()
                and self.expected_frames == self.total_frames
            ):
                ret, _ = cap.read()
                if ret:
                    raise RuntimeError(
                        f"Video reports {self.expected_frames} frames but contains "
                        f"additional decodable frames: {self.video_path}"
                    )
        except Exception as error:
            terminal = _ReaderFailure(error)
        finally:
            if cap is not None:
                cap.release()
            self._enqueue(terminal)

    def get_frame(self):
        """Get next frame from buffer. Returns None when done."""
        item = self.frame_queue.get()
        if isinstance(item, _ReaderFailure):
            raise item.error
        return item

    def stop(self):
        """Stop the reader thread."""
        self.stop_event.set()
        # Drain queue to unblock reader thread
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        self.reader_thread.join(timeout=5.0)


def load_video_chunked(video_path, chunk_size=200, max_frames=None, overlap_frames=0):
    """
    Load video in temporal chunks to minimize memory usage. Chunks may OVERLAP by
    `overlap_frames` (for temporal Hanning blending across chunk boundaries).
    Yields (chunk_frames, chunk_global_start_idx, fps, is_last_chunk, total_frames).

    Args:
        chunk_size: frames per chunk
        max_frames: cap on total frames (None = all)
        overlap_frames: frames shared between consecutive chunks (0 = non-overlapping)
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap_frames < chunk_size:
        raise ValueError(
            f"overlap_frames must be in [0, chunk_size), got {overlap_frames}"
        )
    reader = VideoFrameReader(
        video_path,
        buffer_size=min(50, chunk_size),
        max_frames=max_frames,
    )

    try:
        fps = reader.fps
        total_frames = reader.expected_frames
        stride = chunk_size - overlap_frames

        buf = []
        start = 0
        pending = None

        while True:
            frame = reader.get_frame()
            if frame is None:
                has_new_frames = bool(buf) and (
                    pending is None
                    or overlap_frames == 0
                    or len(buf) > overlap_frames
                )
                if pending is not None:
                    chunk, chunk_start = pending
                    yield (
                        chunk,
                        chunk_start,
                        fps,
                        not has_new_frames,
                        total_frames,
                    )
                if has_new_frames:
                    yield (np.array(buf), start, fps, True, total_frames)
                break

            buf.append(frame)
            if len(buf) >= chunk_size:
                if pending is not None:
                    chunk, chunk_start = pending
                    yield (chunk, chunk_start, fps, False, total_frames)
                pending = (np.array(buf), start)
                buf = buf[stride:]
                start += stride

    finally:
        reader.stop()


def prepare_output_path(output_path, extension=".avi"):
    """
    Ensures the output directory exists and handles file extensions.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() != extension:
        out_path = out_path.with_suffix(extension)

    return out_path


class StreamingVideoWriter:
    """Streaming video writer for chunked processing."""
    def __init__(self, output_path, fps, width, height, extension=".avi"):
        self.output_path = prepare_output_path(output_path, extension)
        self.fps = fps
        self.width = width
        self.height = height
        self.frames_written = 0

        if extension == ".avi":
            # Lossless PNG codec for predictions
            fourcc = cv2.VideoWriter_fourcc(*"MPNG")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height), isColor=False)
        else:
            # MP4 with H.264 codec for maximum compatibility (macOS Preview, VS Code, etc.)
            # Try 'avc1' (H.264) first, fall back to 'mp4v' if not available
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height), isColor=True)

            if not self.writer.isOpened():
                print("    Note: H.264 (avc1) codec not available, trying mp4v...")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height), isColor=True)

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")

    def write_prediction_chunk(self, chunk_predictions, threshold):
        """Write prediction chunk (T, H, W) with binary threshold."""
        for t in range(chunk_predictions.shape[0]):
            frame_uint8 = (chunk_predictions[t] > threshold).astype(np.uint8) * 255
            self.writer.write(frame_uint8)
            self.frames_written += 1

    def write_overlay_chunk(self, original_frames, predictions, threshold):
        """Write overlay chunk with predictions overlaid on original frames."""
        for t in range(predictions.shape[0]):
            img = original_frames[t].astype(np.float32)

            red_mask = np.zeros_like(img)
            red_mask[:, :, 0] = 255.0

            alpha = predictions[t][..., np.newaxis] > threshold
            blended = (img * (1.0 - alpha) + red_mask * alpha).astype(np.uint8)

            # OpenCV uses BGR
            self.writer.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            self.frames_written += 1

    def close(self):
        """Close the video writer."""
        self.writer.release()
        print(f"  Saved {self.frames_written} frames to: {self.output_path.name}")

class DensityMapAccumulator:
    """Accumulates detection counts for density map without storing all frames."""
    def __init__(self, height, width):
        self.density_map = np.zeros((height, width), dtype=np.float32)
        self.total_frames = 0

    def add_chunk(self, chunk_predictions, threshold):
        """Add a chunk of predictions to the density map."""
        binary_preds = (chunk_predictions > threshold).astype(np.float32)
        self.density_map += np.sum(binary_preds, axis=0)
        self.total_frames += chunk_predictions.shape[0]

    def save(self, output_path):
        """Save the accumulated density map."""
        max_val = self.density_map.max()

        plt.figure(figsize=(10, 8))
        plt.imshow(self.density_map, cmap='viridis')
        plt.colorbar(label='Number of frames detected')
        plt.title(f"Detection Density Map\nMax detections: {max_val:.0f} / {self.total_frames} frames")
        plt.xlabel("Cross-shore (W)")
        plt.ylabel("Alongshore (H)")

        density_out_path = Path(output_path).with_name(f"{Path(output_path).stem}_density.png")
        plt.savefig(density_out_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved density map: {density_out_path.name}")


def _validate_output_frame_counts(
    expected_frames,
    processed_frames,
    prediction_writer,
    overlay_writer,
    density_accumulator,
):
    counts = {
        "processed": processed_frames,
        "prediction": prediction_writer.frames_written,
        "overlay": overlay_writer.frames_written,
        "density": density_accumulator.total_frames,
    }
    mismatched = {
        name: count for name, count in counts.items() if count != expected_frames
    }
    if mismatched:
        detail = ", ".join(f"{name}={count}" for name, count in mismatched.items())
        raise RuntimeError(
            f"Frame-count mismatch: expected={expected_frames}, {detail}"
        )


def temporal_blend_weights(n, overlap_frames, *, fade_in, fade_out):
    """Return overlap-add weights without attenuating exposed video endpoints."""
    weights = np.ones(n, dtype=np.float32)
    if overlap_frames == 1:
        if fade_in:
            weights[0] = 0.5
        if fade_out:
            weights[-1] = 0.5
    elif overlap_frames > 1:
        hanning = np.hanning(2 * overlap_frames)
        if fade_in:
            weights[:overlap_frames] = hanning[:overlap_frames]
        if fade_out:
            weights[-overlap_frames:] = hanning[overlap_frames:]
    return weights


def infer_valid_footprint(chunk_frames, minimum_black_frames=224, previous=None):
    """Infer a chunk-scale camera footprint without masking short data gaps.

    A pixel is outside the footprint only when it stays exactly black for an
    entire ``minimum_black_frames`` window. Consequently, dropped frames and
    shorter black intervals remain eligible for model prediction. A short
    final chunk reuses the preceding footprint because it cannot establish a
    full-length black run by itself.
    """
    if len(chunk_frames) < minimum_black_frames:
        if previous is not None:
            return previous
        return np.ones(chunk_frames.shape[1:3], dtype=bool)
    return np.any(chunk_frames[:minimum_black_frames] != 0, axis=(0, 3))


def apply_valid_footprint(
    predictions,
    chunk_frames,
    minimum_black_frames=224,
    previous=None,
):
    """Zero only pixels outside the chunk-scale rectification footprint."""
    valid_footprint = infer_valid_footprint(
        chunk_frames,
        minimum_black_frames=minimum_black_frames,
        previous=previous,
    )
    return predictions * valid_footprint[None], valid_footprint


def process_video(
    video_path,
    model,
    device,
    output_path,
    *,
    overlap_y,
    overlap_x,
    detection_threshold,
    patch_size=224,
    batch_size=32,
    n_frames=None,
    chunk_size=200,
    alongshore_sigma=0.0,
    temporal_overlap=0.0,
    max_alongshore=None,
    transect_group_size=1,
):
    """Process video using streaming I/O with chunked timestack format.

    temporal_overlap: fraction of chunk_size that consecutive time-chunks share; their
      soft predictions are Hanning-blended over the overlap to remove chunk-boundary seams.
    max_alongshore: if set, only process the first N alongshore (H) rows (for quick tests).
    overlap_y / overlap_x: cross-shore/time tile overlaps.
    """
    print(f"Processing video: {video_path}")
    print(f"  Chunk size: {chunk_size} frames")
    print(f"  Batch size: {batch_size} patches")
    print(f"  Transect group size: {transect_group_size}")

    if transect_group_size <= 0:
        raise ValueError(
            f"transect_group_size must be positive, got {transect_group_size}"
        )

    pred_writer = None
    overlay_writer = None
    density_accumulator = None
    fps = None
    total_frames = None
    overall_pbar = None

    if not 0 <= temporal_overlap < 1:
        raise ValueError(f"temporal_overlap must be in [0, 1), got {temporal_overlap}")

    # Temporal chunk overlap + Hanning blend removes time-chunk boundary seams.
    overlap_frames = round(temporal_overlap * chunk_size) if temporal_overlap and temporal_overlap > 0 else 0
    if overlap_frames >= chunk_size:
        raise ValueError(
            "temporal_overlap rounds to a full chunk; reduce temporal_overlap"
        )
    stride = chunk_size - overlap_frames
    acc_sum, acc_w, acc_raw = {}, {}, {}   # global_frame_idx -> soft-sum (H,W) / weight / raw frame
    flush_ptr = [0]
    valid_footprint = None

    def _flush(until):
        while flush_ptr[0] < until:
            gi = flush_ptr[0]
            pred = (acc_sum[gi] / max(acc_w[gi], 1e-8))[None]   # (1,H,W) blended soft prob
            raw = acc_raw[gi][None]                              # (1,H,W,C)
            pred_writer.write_prediction_chunk(pred, threshold=detection_threshold)
            overlay_writer.write_overlay_chunk(raw, pred, threshold=detection_threshold)
            density_accumulator.add_chunk(pred, threshold=detection_threshold)
            del acc_sum[gi], acc_w[gi], acc_raw[gi]
            overall_pbar.update(1)
            flush_ptr[0] += 1

    for chunk_idx, (chunk_frames, start_frame_idx, chunk_fps, is_last, total_frames_in_video) in enumerate(
        load_video_chunked(video_path, chunk_size=chunk_size, max_frames=n_frames, overlap_frames=overlap_frames)
    ):
        if max_alongshore:
            chunk_frames = chunk_frames[:, :max_alongshore]
        if fps is None:
            fps = chunk_fps
            total_frames = total_frames_in_video
            # Create overall progress bar (disabled when output isn't a TTY, e.g. nohup -> file,
            # so the log isn't spammed with \r progress lines)
            overall_pbar = tqdm(
                total=total_frames,
                desc="Overall Progress",
                unit="frames",
                position=0,
                leave=True,
                disable=not sys.stdout.isatty(),
            )

        T, H, W, C = chunk_frames.shape

        if pred_writer is None:
            pred_writer = StreamingVideoWriter(output_path, fps, W, H, extension=".avi")

            overlay_path = str(Path(output_path).with_suffix("")) + "_overlay.mp4"
            overlay_writer = StreamingVideoWriter(overlay_path, fps, W, H, extension=".mp4")

            density_accumulator = DensityMapAccumulator(H, W)
            overall_pbar.write("Initialized output streams:")
            overall_pbar.write(f"  - Predictions: {pred_writer.output_path.name}")
            overall_pbar.write(f"  - Overlay: {overlay_writer.output_path.name}")

        # Process this chunk's timestacks. Grouping transects fills model batches
        # across row boundaries; group_size=1 retains the original path.
        if transect_group_size == 1:
            chunk_predictions = []
            for yFRF in tqdm(
                range(H),
                desc=f"Chunk {chunk_idx} (frames {start_frame_idx}-{start_frame_idx + T - 1})",
                unit="transect",
                position=1,
                leave=False,
                disable=not sys.stdout.isatty(),
            ):
                ts_slice = chunk_frames[:, yFRF, ::-1, :].transpose(1, 0, 2)
                if not ts_slice.any():
                    pred = np.zeros(ts_slice.shape[:2], dtype=np.float32)
                else:
                    pred = predict_tiled(
                        model,
                        ts_slice.astype(np.float32) / 255.0,
                        patch_size=patch_size,
                        overlap_y=overlap_y,
                        overlap_x=overlap_x,
                        batch_size=batch_size,
                        device=device,
                        skip_zero_patches=True,
                    )
                chunk_predictions.append(pred)
                if yFRF > 0 and yFRF % 100 == 0:
                    cleanup_memory(device)
            chunk_predictions = np.asarray(chunk_predictions)
        else:
            chunk_predictions = np.zeros((H, W, T), dtype=np.float32)
            active_rows = np.flatnonzero(np.any(chunk_frames, axis=(0, 2, 3)))
            progress = tqdm(
                total=len(active_rows),
                desc=f"Chunk {chunk_idx} (frames {start_frame_idx}-{start_frame_idx + T - 1})",
                unit="transect",
                position=1,
                leave=False,
                disable=not sys.stdout.isatty(),
            )
            for group_start in range(0, len(active_rows), transect_group_size):
                rows = active_rows[group_start : group_start + transect_group_size]
                timestacks = chunk_frames[:, rows, ::-1, :].transpose(1, 2, 0, 3)
                chunk_predictions[rows] = predict_tiled_many(
                    model,
                    timestacks.astype(np.float32) / 255.0,
                    patch_size=patch_size,
                    overlap_y=overlap_y,
                    overlap_x=overlap_x,
                    batch_size=batch_size,
                    device=device,
                    skip_zero_patches=True,
                )
                progress.update(len(rows))
            progress.close()
        # Optional alongshore (H, axis 0) Gaussian smooth of the RAW probabilities to enforce
        # continuity across independently-predicted transects (reduces alongshore streakiness).
        if alongshore_sigma and alongshore_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            chunk_predictions = gaussian_filter1d(
                chunk_predictions, sigma=alongshore_sigma, axis=0, mode="reflect"
            )
        chunk_pred_frames = np.transpose(chunk_predictions, (2, 0, 1))[:, :, ::-1]  # (T, H, W) soft
        # The rectified videos encode pixels outside the valid camera
        # footprint as exactly black. Tiling and especially alongshore
        # smoothing can otherwise leak nearby probabilities into those pixels.
        chunk_pred_frames, valid_footprint = apply_valid_footprint(
            chunk_pred_frames,
            chunk_frames,
            minimum_black_frames=patch_size,
            previous=valid_footprint,
        )

        # Temporal overlap-add: accumulate soft probs with a Hanning weight, then flush
        # finalized frames (those no future chunk will touch). overlap_frames=0 -> 1 chunk/frame.
        tw = temporal_blend_weights(
            T,
            overlap_frames,
            fade_in=start_frame_idx > 0,
            fade_out=not is_last,
        )
        for i in range(T):
            gi = start_frame_idx + i
            contrib = chunk_pred_frames[i] * tw[i]
            if gi in acc_sum:
                acc_sum[gi] += contrib
                acc_w[gi] += float(tw[i])
            else:
                acc_sum[gi] = contrib
                acc_w[gi] = float(tw[i])
                acc_raw[gi] = chunk_frames[i]
        _flush(start_frame_idx + T if is_last else start_frame_idx + stride)
        if not sys.stdout.isatty():   # coarse progress for the log (tqdm is disabled there)
            print(f"  chunk {chunk_idx + 1}: frames {start_frame_idx}-{start_frame_idx + T - 1}"
                  f" / {total_frames} done", flush=True)

        del chunk_frames, chunk_predictions, chunk_pred_frames
        cleanup_memory(device)

    # Final flush of any remaining buffered frames (safety; is_last normally drains them)
    if acc_sum:
        _flush(max(acc_sum.keys()) + 1)

    if overall_pbar:
        overall_pbar.close()

    if pred_writer:
        print("\nFinalizing outputs...")
        try:
            pred_writer.close()
        finally:
            overlay_writer.close()
        _validate_output_frame_counts(
            total_frames,
            flush_ptr[0],
            pred_writer,
            overlay_writer,
            density_accumulator,
        )
        density_accumulator.save(output_path)

    cleanup_memory(device)

    print("Video processing complete!")


def process_directory(input_dir, model, device, output_dir, **kwargs):
    """Process all videos in a directory with memory cleanup between videos."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = ['.avi'] # [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))

    video_files = sorted(video_files)  # Process in consistent order
    if not video_files:
        raise FileNotFoundError(f"No input videos found in {input_path}")
    print(f"Found {len(video_files)} video files")

    videos_to_process = []
    for video_file in video_files:
        # Density png is written last, so it marks a FULLY-complete video (a bare _pred.avi
        # may be a partial from an interrupted run -> redo it).
        density_file = output_path / f"{video_file.stem}_pred_density.png"
        if density_file.exists():
            print(f"Skipping {video_file.name} (already complete)")
        else:
            videos_to_process.append(video_file)

    print(f"Processing {len(videos_to_process)} videos (skipped {len(video_files) - len(videos_to_process)} already completed)")
    print()

    failures = []
    for idx, video_file in enumerate(videos_to_process):
        print(f"\n{'='*80}")
        print(f"Video {idx + 1}/{len(videos_to_process)}: {video_file.name}")
        print(f"{'='*80}")

        output_file = output_path / f"{video_file.stem}_pred.avi"

        try:
            process_video(video_file, model, device, output_file, **kwargs)
        except Exception as e:
            failures.append(video_file.name)
            print(f"ERROR processing {video_file.name}: {e}")
            traceback.print_exc()
        finally:
            cleanup_memory(device)

        print()

    if failures:
        raise RuntimeError(f"Prediction failed for {len(failures)} video(s): {', '.join(failures)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run production breaking-wave crest inference on one video or a directory."
    )
    parser.add_argument("config", type=Path, help="prediction YAML")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="override the config device (default: auto-detect)",
    )
    parser.add_argument("--model-path", help="override the external checkpoint path")
    parser.add_argument("--input-path", help="override the input video or directory")
    parser.add_argument("--output-path", help="override the output file or directory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="resolve paths and strictly load the checkpoint without processing video",
    )
    return parser.parse_args(argv)


def load_prediction_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    required = {
        "model_path",
        "input_path",
        "output_path",
        "overlap_y",
        "overlap_x",
        "detection_threshold",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Prediction config is missing keys: {missing}")
    unknown = sorted(
        set(config)
        - {
            "model_path", "input_path", "output_path", "device", "patch_size",
            "overlap_y", "overlap_x", "batch_size",
            "transect_group_size", "chunk_size", "alongshore_smooth",
            "detection_threshold", "temporal_overlap", "n_frames",
            "max_alongshore",
        }
    )
    if unknown:
        raise ValueError(f"Unknown prediction config keys: {unknown}")
    return config


def main(argv=None):
    args = parse_args(argv)
    config = load_prediction_config(args.config)
    for key in ("model_path", "input_path", "output_path", "device"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    print("Loading production model...")
    model_path = dunex_paths.resolve(config["model_path"])
    model, checkpoint_config = load_model(model_path, device=device)
    seed_inference(int(checkpoint_config["seed"]), device)

    patch_size = int(config.get("patch_size", checkpoint_config["image_size"]))
    chunk_size = int(config.get("chunk_size", patch_size))
    if patch_size != int(checkpoint_config["image_size"]):
        raise ValueError(
            f"patch_size must match the checkpoint image size "
            f"({checkpoint_config['image_size']})"
        )
    if chunk_size != patch_size:
        raise ValueError(
            "chunk_size must equal patch_size; temporal_overlap controls overlap "
            "between complete temporal chunks"
        )
    overlap_x = int(config["overlap_x"])
    if overlap_x != 0:
        raise ValueError("overlap_x must be 0 for the production temporal policy")
    detection_threshold = float(config["detection_threshold"])
    if detection_threshold != PRODUCTION_THRESHOLD:
        raise ValueError(
            f"The production operating point is {PRODUCTION_THRESHOLD}; "
            f"got {detection_threshold}"
        )
    overlap_y = int(config["overlap_y"])
    batch_size = int(config.get("batch_size", 32))
    transect_group_size = int(config.get("transect_group_size", 64))
    temporal_overlap = float(config.get("temporal_overlap", 0.10))
    alongshore_sigma = float(config.get("alongshore_smooth", 2.0))
    if not 0 <= overlap_y < patch_size:
        raise ValueError("overlap_y must be in [0, patch_size)")
    if batch_size <= 0 or transect_group_size <= 0:
        raise ValueError("batch_size and transect_group_size must be positive")
    if not 0 <= temporal_overlap < 1:
        raise ValueError("temporal_overlap must be in [0, 1)")
    if alongshore_sigma < 0:
        raise ValueError("alongshore_smooth must be non-negative")

    input_path = dunex_paths.resolve(config["input_path"])
    output_path = str(dunex_paths.resolve(config["output_path"]))
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Threshold: {detection_threshold}")
    if args.validate_only:
        print("Configuration and checkpoint validation complete.")
        return

    common = {
        "patch_size": patch_size,
        "batch_size": batch_size,
        "n_frames": config.get("n_frames"),
        "chunk_size": chunk_size,
        "alongshore_sigma": alongshore_sigma,
        "detection_threshold": detection_threshold,
        "temporal_overlap": temporal_overlap,
        "max_alongshore": config.get("max_alongshore"),
        "overlap_y": overlap_y,
        "overlap_x": overlap_x,
        "transect_group_size": transect_group_size,
    }
    if input_path.is_file():
        process_video(input_path, model, device, output_path, **common)
    elif input_path.is_dir():
        process_directory(input_path, model, device, output_path, **common)


if __name__ == "__main__":
    main()
