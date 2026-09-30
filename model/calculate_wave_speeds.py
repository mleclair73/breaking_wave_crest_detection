#!/usr/bin/env python3
"""
Full pipeline for calculating wave speeds from prediction videos.
Parallelized for maximum throughput.

Usage:
    python calculate_wave_speeds.py <input_video_or_folder> <output_folder>
"""

import sys
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from common import datetime_utils as _datetime_utils
from scipy.ndimage import label
from scipy.signal import savgol_filter
from tqdm import tqdm


def build_per_pixel_speed_map(height, width, results):
    """
    Build a per-pixel speed map where each pixel gets the local speed
    of the wave track that passed through it.

    Maps local track speeds back to ALL constituent points of each blob,
    using the local speed at each point in the track.

    Returns:
        speed_sum: Sum of speeds at each pixel
        speed_sq_sum: Sum of squared speeds at each pixel
        counts: Number of observations at each pixel
        crest_count: Number of distinct crests at each pixel
        speed_lists: Sparse {(y, x): [speeds]} of the individual speeds at each
            pixel, needed for the median (running sums alone can't produce it)
    """
    speed_sum = np.zeros((height, width), dtype=float)
    speed_sq_sum = np.zeros((height, width), dtype=float)
    counts = np.zeros((height, width), dtype=float)
    crest_count = np.zeros((height, width), dtype=float)   # distinct crests per pixel
    speed_lists = {}                                       # only pixels waves pass through

    for res in results:
        y = res['y_idx']
        for track_entry, local_speeds in res['track_data']:
            visited = set()                                # pixels this crest has touched
            # Each track_entry is a list of (t, centroid, blob_pixels)
            # local_speeds is an array with one speed per point in track
            for i, (_, _, blob_pixels) in enumerate(track_entry):
                speed = local_speeds[i]
                for xi in blob_pixels:
                    if not (0 <= xi < width):
                        continue
                    if xi not in visited:
                        crest_count[y, xi] += 1
                        visited.add(xi)
                    if not np.isfinite(speed):
                        continue
                    speed_sum[y, xi] += speed
                    speed_sq_sum[y, xi] += speed ** 2
                    counts[y, xi] += 1
                    speed_lists.setdefault((y, xi), []).append(speed)

    return speed_sum, speed_sq_sum, counts, crest_count, speed_lists


def calculate_trimmed_frame_count(video_frames):
    """
    Calculate the number of frames after trimming all-zero frames from start and end.

    Returns:
        trimmed_count: Number of non-empty frames
        first_nonzero: Index of first non-zero frame
        last_nonzero: Index of last non-zero frame
    """
    T = video_frames.shape[0]

    first_nonzero = 0
    for t in range(T):
        if video_frames[t].any():
            first_nonzero = t
            break
    else:
        # All frames are zero
        return 0, 0, 0

    last_nonzero = T - 1
    for t in range(T - 1, -1, -1):
        if video_frames[t].any():
            last_nonzero = t
            break

    trimmed_count = last_nonzero - first_nonzero + 1
    return trimmed_count, first_nonzero, last_nonzero


def create_netcdf(height, width, pixel_maps, output_path, video_path, video_frames=None, fps=None):
    """
    Create NetCDF file with average squared speed per pixel.

    Coordinates (integer FRF metres, 1 pixel per metre):
        - xFRF: 0 .. 499 (cross-shore, horizontal axis)
        - yFRF: 1500 (top) .. -99 (bottom) (alongshore, vertical axis)

    `pixel_maps` is the tuple returned by build_per_pixel_speed_map.
    """
    speed_sum, speed_sq_sum, counts, crest_count, speed_lists = pixel_maps

    if video_frames is not None:
        total_frames = video_frames.shape[0]
        trimmed_frames, first_frame, last_frame = calculate_trimmed_frame_count(video_frames)
        if fps is not None and fps > 0:
            video_duration = trimmed_frames / fps
        else:
            video_duration = np.nan
    else:
        total_frames = 0
        trimmed_frames = 0
        video_duration = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_sq_speed = speed_sq_sum / counts
        mean_sq_speed[counts == 0] = np.nan
        mean_speed = speed_sum / counts
        mean_speed[counts == 0] = np.nan

    median_speed = np.full((height, width), np.nan)
    for (y, xi), speeds in speed_lists.items():
        median_speed[y, xi] = np.median(speeds)

    # Integer FRF metres, exactly 1 m / pixel (linspace endpoints gave ~1.002 m spacing).
    xFRF = np.arange(width)                  # 0 .. 499  (width == 500)
    yFRF = np.arange(1500, 1500 - height, -1)  # 1500 (top) .. -99 (bottom)  (height == 1600)

    dt = _datetime_utils.parse_datetime_from_filename(video_path.name)

    mean_sq_speed = mean_sq_speed[np.newaxis, :, :]
    mean_speed = mean_speed[np.newaxis, :, :]
    median_speed = median_speed[np.newaxis, :, :]
    counts = counts[np.newaxis, :, :]
    crest_count = crest_count[np.newaxis, :, :]

    ds = xr.Dataset(
        {
            'mean_squared_speed': (['time', 'yFRF', 'xFRF'], mean_sq_speed, {
                'units': 'm^2/s^2',
                'long_name': 'Mean Squared Wave Speed',
                'description': 'Average of squared individual wave speeds at each pixel'
            }),
            'mean_speed': (['time', 'yFRF', 'xFRF'], mean_speed, {
                'units': 'm/s',
                'long_name': 'Mean Wave Speed',
                'description': 'Average individual wave speed at each pixel'
            }),
            'median_speed': (['time', 'yFRF', 'xFRF'], median_speed, {
                'units': 'm/s',
                'long_name': 'Median Wave Speed',
                'description': 'Median of individual wave speeds at each pixel'
            }),
            'observation_count': (['time', 'yFRF', 'xFRF'], counts, {
                'units': '1',
                'long_name': 'Observation Count',
                'description': 'Number of wave detections at each pixel'
            }),
            'crest_count': (['time', 'yFRF', 'xFRF'], crest_count, {
                'units': '1',
                'long_name': 'Crest Count',
                'description': 'Number of unique wave tracks observed at each pixel'
            }),

            'video_duration': (['time'], [video_duration], {
                'units': 's',
                'long_name': 'Video Duration (Trimmed)',
                'description': 'Duration of video after trimming all-zero frames from start and end'
            }),
            'total_frames': (['time'], np.array([total_frames], dtype=np.int32), {
                'units': '1',
                'long_name': 'Total Frame Count',
                'description': 'Total number of frames in original video'
            }),
            'trimmed_frames': (['time'], np.array([trimmed_frames], dtype=np.int32), {
                'units': '1',
                'long_name': 'Trimmed Frame Count',
                'description': 'Number of frames after trimming all-zero frames from start and end'
            }),
            'fps': (['time'], np.array([fps if fps is not None else np.nan], dtype=np.float32), {
                'units': 'Hz',
                'long_name': 'Video Frame Rate',
                'description': 'Frame rate used for this video (1 Hz for 2021-09-19, 2 Hz otherwise)'
            }),
        },
        coords={
            'time': (['time'], [np.datetime64(dt)], {
                'long_name': 'Video Start Time',
                'axis': 'T'
            }),
            'xFRF': (['xFRF'], xFRF, {
                'units': 'm',
                'long_name': 'Cross-shore Position (FRF)',
                'axis': 'X'
            }),
            'yFRF': (['yFRF'], yFRF, {
                'units': 'm',
                'long_name': 'Alongshore Position (FRF)',
                'axis': 'Y'
            }),
        },
        attrs={
            'title': 'Wave Speed Map',
            'source': video_path.name,
            'institution': 'FRF',
            'history': f'Created {datetime.now().isoformat()}',
            'time_coverage_start': dt.isoformat() + 'Z',
        }
    )

    encoding = {
        'mean_squared_speed': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'mean_speed': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'median_speed': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'observation_count': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'crest_count': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'video_duration': {'dtype': 'float64'},
        'total_frames': {'dtype': 'int32'},
        'trimmed_frames': {'dtype': 'int32'},
        'fps': {'dtype': 'float32'},
        'time': {'units': 'seconds since 1970-01-01T00:00:00Z', 'dtype': 'float64'},
        'xFRF': {'dtype': 'float32'},
        'yFRF': {'dtype': 'float32'},
    }
    ds.to_netcdf(output_path, encoding=encoding, format='NETCDF4')
    print(f"  NetCDF saved: {output_path.name} (trimmed: {trimmed_frames}/{total_frames} frames, {video_duration:.1f}s)")

    return ds


def load_prediction_video(video_path, n_frames=None):
    """Load prediction video as binary masks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_frames <= 0:
            raise ValueError(f"Video reports no frames: {video_path}")
        if n_frames is not None and n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        expected_frames = (
            min(reported_frames, n_frames)
            if n_frames is not None
            else reported_frames
        )
        frames = []

        for frame_idx in range(expected_frames):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(
                    f"Decoded {frame_idx}/{expected_frames} expected frames from "
                    f"{video_path}"
                )

            binary = (frame[:, :, 0] > 127).astype(np.uint8)
            frames.append(binary)

        if expected_frames == reported_frames:
            ret, _ = cap.read()
            if ret:
                raise RuntimeError(
                    f"Video reports {reported_frames} frames but contains additional "
                    f"decodable frames: {video_path}"
                )
    finally:
        cap.release()

    return np.array(frames), fps, (width, height)


def extract_timestack(video_frames, y_transect):
    """Extract timestack for a specific alongshore position."""
    return video_frames[:, y_transect, :]


def compute_local_speeds_finite_diff(times, positions, max_speed=20):
    """
    Compute local speeds using finite differences.
    - Central difference for interior points
    - Forward/backward difference for endpoints

    Returns array of speeds (m/s) at each point.
    """
    n = len(times)
    if n < 2:
        return np.full(n, np.nan)

    speeds = np.full(n, np.nan)

    for i in range(n):
        if i == 0:
            # Forward difference
            dt = times[1] - times[0]
            dx = positions[1] - positions[0]
        elif i == n - 1:
            # Backward difference
            dt = times[-1] - times[-2]
            dx = positions[-1] - positions[-2]
        else:
            # Central difference
            dt = times[i + 1] - times[i - 1]
            dx = positions[i + 1] - positions[i - 1]

        if abs(dt) > 1e-10:
            speed = -dx / dt  # Negative because waves move toward shore (decreasing x)
            if 0 < speed < max_speed:
                speeds[i] = speed

    return speeds


def compute_local_speeds_savgol(times, positions, max_speed=20, window=5, polyorder=2):
    """
    Compute local speeds using Savitzky-Golay derivative.

    Fits a polynomial locally and differentiates analytically, which
    smooths noise while computing derivatives - much better for noisy data
    than raw finite differences.

    Args:
        times: Array of time values (s)
        positions: Array of position values (m)
        max_speed: Maximum valid speed (m/s), speeds outside (0, max_speed) are NaN
        window: Window length for Savitzky-Golay filter (must be odd, >= polyorder+2)
        polyorder: Polynomial order for fitting (2 = quadratic is typical)

    Returns:
        Array of speeds (m/s) at each point.
    """
    n = len(times)
    if n < 2:
        return np.full(n, np.nan)

    # Need at least window points for savgol, and window must be odd
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2:
        # Fall back to finite differences if track too short
        return compute_local_speeds_finite_diff(times, positions, max_speed)

    # Compute dt - assume uniform spacing (use mean dt)
    dt_mean = np.mean(np.diff(times))
    if abs(dt_mean) < 1e-10:
        return np.full(n, np.nan)

    # Savitzky-Golay derivative: deriv=1 gives first derivative
    # delta parameter scales the derivative by 1/delta (so we use dt_mean)
    dx_dt = savgol_filter(positions, window, polyorder, deriv=1, delta=dt_mean)

    # Speed is negative derivative (waves move toward shore = decreasing x)
    speeds = -dx_dt

    # Apply speed limits
    speeds[(speeds <= 0) | (speeds >= max_speed)] = np.nan

    return speeds


# Frames on each side (in time) of a crest-merge centroid whose speed is blanked.
MERGE_BLANK = 2


def _component_subtracks(comp_mask, times_present, gate_px, max_gap=3):
    """Split one CC component into sub-tracks via per-row 1D blobs + greedy nearest-neighbour
    association. Separates crossing / parallel crests that the row-mean centroid would merge
    into a single phantom mid-line, while keeping the component's diagonal-connectivity
    gap-bridging. (Enabled by `split_crossings`; option A in the brainstorm — a cheap,
    velocity-free tracker, not the full Kalman/Hungarian version.)

    Returns a list of (pts, merge_idx) per sub-track, where pts is [(t, centroid, blob_px), ...]
    and merge_idx is the set of pts-indices where the assigned blob was *contested* by >=2
    tracks (a crest merge): there the centroid sits between the two crests, not on either, so
    the local speed there (and its savgol neighbours) is bogus and gets blanked downstream.
    """
    active, finished = [], []   # active track = {'pts':[...], 'merge':[...], 'x':last c, 't':last t}
    for t in times_present:
        lab, n = label(comp_mask[t, :])                  # 1D connected blobs in this row
        blobs = [np.where(lab == b)[0] for b in range(1, n + 1)]
        cents = [float(px.mean()) for px in blobs]
        cands = []                                       # candidate (dist, track, blob) matches
        for ai, tr in enumerate(active):
            gap = t - tr['t']
            if gap > max_gap:
                continue
            for bi, c in enumerate(cents):
                d = abs(c - tr['x'])
                if d <= gate_px * max(gap, 1):           # time-scaled position gate
                    cands.append((d, ai, bi))
        cands.sort()
        used_a, used_b = set(), set()
        blob_winner = {}                                 # blob -> (track, point_index) that took it
        for d, ai, bi in cands:                          # greedy nearest-neighbour assignment
            if ai in used_a or bi in used_b:
                continue
            tr = active[ai]
            blob_winner[bi] = (ai, len(tr['pts']))
            tr['pts'].append((t, cents[bi], blobs[bi].tolist()))
            tr['x'], tr['t'] = cents[bi], t
            used_a.add(ai); used_b.add(bi)
        # A merge this row = an in-gate track left *starved* (its nearest blob was taken by
        # another track because two crests share one blob). The winner that took it now sits on
        # the phantom midpoint -> flag the winner's point (only the continuing track, never the
        # clean one that simply gaps here).
        for ai in {a for _, a, _ in cands} - used_a:
            wanted = min((d, bi) for d, a, bi in cands if a == ai)[1]
            if wanted in blob_winner:
                wtr, widx = blob_winner[wanted]
                active[wtr]['merge'].append(widx)
        for bi, c in enumerate(cents):                   # unmatched blobs start new tracks
            if bi not in used_b:
                active.append({'pts': [(t, c, blobs[bi].tolist())], 'merge': [], 'x': c, 't': t})
        keep = []                                        # retire tracks idle > max_gap frames
        for tr in active:
            (finished if t - tr['t'] > max_gap else keep).append(tr)
        active = keep
    finished += active
    return [(tr['pts'], set(tr['merge'])) for tr in finished]


def track_centroids_simple(timestack, dt=0.5, dx=1.0, min_track_duration=2.5,
                           speed_method='savgol', savgol_window=5, savgol_polyorder=2,
                           split_crossings=True, gate_max_speed=20.0):
    """
    Track wave blobs using 2D connected component labeling on the timestack.

    Waves form continuous diagonal streaks in the timestack, so we label
    the entire 2D array and extract tracks from each connected component.
    This is more robust than frame-by-frame greedy matching.

    Args:
        timestack: 2D array (T, W) of binary mask values
        dt: Time step in seconds
        dx: Spatial resolution in meters/pixel
        min_track_duration: Minimum duration in seconds for a valid track,
            converted to frames via dt so 1 Hz and 2 Hz videos apply the same
            physical cutoff (floor of 3 frames; short tracks fall back to
            finite differences in the savgol path)
        speed_method: 'savgol' (Savitzky-Golay, smoother) or 'finite_diff' (raw)
        savgol_window: Window size for Savitzky-Golay (odd integer, default 5)
        savgol_polyorder: Polynomial order for Savitzky-Golay (default 2)

    Returns:
        track_data: List of tuples (track_with_pixels, local_speeds)
            where track_with_pixels is [(t, centroid, blob_pixels), ...]
            local_speeds is array of speeds at each point in the track
        mean_speed: Scalar mean for the transect
    """
    label_struct = np.array([
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
    ], dtype=bool)

    labeled_timestack, n_components = label(timestack, structure=label_struct)

    # Seconds -> frames: at 2 Hz (dt=0.5) the 2.5 s default gives the original 5 frames;
    # at 1 Hz (dt=1.0) it gives 3 frames instead of a silently stricter 5 s cutoff.
    min_track_length = max(3, int(np.ceil(min_track_duration / dt)))

    # Max px a crest moves per frame — gate for the optional crossing-split NN association.
    gate_px = gate_max_speed * dt / dx

    tracks = []
    track_merges = []   # per-track set of pts-indices flagged as crest-merge centroids
    # Crossing-split firing stats: components the split path turned into >=2 kept sub-tracks
    # (i.e. where the flag actually changed the output) out of all components it kept.
    n_components_split = 0
    n_components_kept = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = (labeled_timestack == comp_id)
        times_present = np.where(comp_mask.any(axis=1))[0]

        if len(times_present) < min_track_length:
            continue

        if split_crossings:
            # Split multi-blob rows + NN-associate -> one sub-track per crest (no phantom mean).
            kept = [(pts, merge) for pts, merge in _component_subtracks(comp_mask, times_present, gate_px)
                    if len(pts) >= min_track_length]
            tracks.extend(pts for pts, _ in kept)
            track_merges.extend(merge for _, merge in kept)
            if len(kept) >= 1:
                n_components_kept += 1
            if len(kept) >= 2:
                n_components_split += 1
        else:
            # Original: one row-mean centroid per frame (averages crossings into a phantom).
            track = []
            for t in times_present:
                positions = np.where(comp_mask[t, :])[0]
                if len(positions) > 0:
                    track.append((t, float(np.mean(positions)), positions.tolist()))
            if len(track) >= min_track_length:
                tracks.append(track)
                track_merges.append(set())

    track_data = []
    all_speeds = []
    for track, merges in zip(tracks, track_merges):
        times = np.array([t * dt for t, _, _ in track])
        positions = np.array([centroid * dx for _, centroid, _ in track])

        if speed_method == 'savgol':
            local_speeds = compute_local_speeds_savgol(
                times, positions, window=savgol_window, polyorder=savgol_polyorder
            )
        else:  # 'finite_diff'
            local_speeds = compute_local_speeds_finite_diff(times, positions)

        # Blank the speed within +/-MERGE_BLANK frames of a crest merge on this (continuing)
        # track: at the merge the centroid is the midpoint between two crests, so it (and its
        # savgol neighbours) gives a bogus speed jump. Window is in TIME (frames), not list
        # index, since the tracker bridges gaps. This avoids assigning a speed to the
        # midpoint of a merged crest.
        if merges:
            frames = np.array([t for t, _, _ in track])
            for mi in merges:
                local_speeds[np.abs(frames - frames[mi]) <= MERGE_BLANK] = np.nan

        track_data.append((track, local_speeds))
        valid_speeds = local_speeds[np.isfinite(local_speeds)]
        all_speeds.extend(valid_speeds.tolist())

    split_stats = (n_components_split, n_components_kept)
    return track_data, np.mean(all_speeds) if all_speeds else np.nan, split_stats

def process_transect(args):
    y_idx, timestack, dt, dx, speed_method, savgol_window, savgol_polyorder, split_crossings = args
    track_data, _, split_stats = track_centroids_simple(
        timestack, dt, dx,
        speed_method=speed_method,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        split_crossings=split_crossings,
    )
    return {
        'y_idx': y_idx,
        'track_data': track_data,  # List of ([(t, centroid, blob_pixels), ...], speed)
        'split_stats': split_stats,  # (n_components_split>=2, n_components_kept)
    }
        
def create_mean_speed_map(height, width, pixel_maps, output_path, vmin, vmax):
    """
    Creates a 2D spatial map with fixed aspect ratio 1.
    Y-axis (yFRF): 1500 at top, -99 at bottom. X-axis (xFRF): 0-499.

    Maps track speeds back to ALL constituent points of each blob.
    `pixel_maps` is the tuple returned by build_per_pixel_speed_map.
    """
    speed_sum, _, counts, _, _ = pixel_maps

    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_map = speed_sum / counts
        pixel_map[counts == 0] = np.nan

    plt.figure(figsize=(10, 12))

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    current_cmap = plt.cm.jet.copy()
    current_cmap.set_bad(color='white')

    im = plt.pcolormesh(X, Y, pixel_map, cmap=current_cmap, 
                        vmin=vmin, vmax=vmax, shading='auto')
    
    ax = plt.gca()
    
    ax.set_aspect(1)
    ax.set_xlim(0, width - 1)
    
    # Match the NetCDF: pixel 0 is yFRF 1500 and yFRF decreases by 1 m per row.
    pixel_ticks = np.arange(0, height + 1, 200)
    yFRF_labels = 1500 - pixel_ticks
    ax.set_yticks(pixel_ticks)
    ax.set_yticklabels(yFRF_labels)

    plt.colorbar(im, label='Local Wave Speed (m/s)', shrink=0.6)
    plt.title('Per-Pixel Mean Wave Speed Map')
    plt.xlabel('Cross-shore Position (xFRF, m)')
    plt.ylabel('Alongshore Position (yFRF, m)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Map saved: {output_path.name} (yFRF: 1500 at top to {1501 - height} at bottom)")


def save_raw_trajectory_data(results, output_path, dt, dx):
    """
    Save raw trajectory data to CSV for analyzing speed variation along tracks.

    Columns:
        - track_id: Unique identifier for each track
        - y_idx: Alongshore position (pixel index, 0=top)
        - yFRF: Alongshore position in FRF coordinates (m)
        - position_in_track: Index within the track (0=start)
        - frame: Frame number
        - time_s: Time in seconds
        - x_centroid: Cross-shore centroid position (pixel)
        - xFRF: Cross-shore position in FRF coordinates (m)
        - local_speed: Local speed at this point (m/s)
        - track_length: Total number of points in this track
    """
    rows = []
    track_id = 0

    for res in results:
        y_idx = res['y_idx']
        yFRF = 1500 - y_idx

        for track_entry, local_speeds in res['track_data']:
            track_length = len(track_entry)

            for i, (frame, centroid, blob_pixels) in enumerate(track_entry):
                speed = local_speeds[i]
                time_s = frame * dt
                xFRF = centroid * dx  # Cross-shore position in meters

                rows.append({
                    'track_id': track_id,
                    'y_idx': y_idx,
                    'yFRF': yFRF,
                    'position_in_track': i,
                    'frame': frame,
                    'time_s': time_s,
                    'x_centroid': centroid,
                    'xFRF': xFRF,
                    'local_speed': speed if np.isfinite(speed) else np.nan,
                    'track_length': track_length,
                })

            track_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Raw trajectories saved: {output_path.name} ({len(df)} points, {track_id} tracks)")

    return df


def process_video(video_path, output_dir, dx=1.0, n_workers=None,
                  speed_method='savgol', savgol_window=5, savgol_polyorder=2,
                  save_videos=False, split_crossings=True):
    """
    Process a single video: calculate speeds and generate spatial maps.

    Args:
        video_path: Path to input prediction video
        output_dir: Directory for output files
        dx: Spatial resolution in meters/pixel
        n_workers: Number of parallel workers (default: cpu_count - 1)
        speed_method: 'savgol' (Savitzky-Golay, smoother) or 'finite_diff' (raw)
        savgol_window: Window size for Savitzky-Golay (odd integer, default 5)
        savgol_polyorder: Polynomial order for Savitzky-Golay (default 2)
        save_videos: If True, save AVI/MP4 speed videos (slow, default False)

    Time resolution (dt) is automatically determined based on video date:
    - September 19, 2021: 1Hz (dt=1.0s)
    - All other dates: 2Hz (dt=0.5s)
    """
    print(f"\nProcessing: {video_path.name}")

    video_dt = _datetime_utils.parse_datetime_from_filename(video_path.name)
    dt, fps_hz = _datetime_utils.get_time_resolution_for_date(video_dt)
    print(f"  Date: {video_dt:%Y-%m-%d}, Frame rate: {fps_hz} Hz (dt={dt}s)")
    print(f"  Speed method: {speed_method}" + (f" (window={savgol_window}, order={savgol_polyorder})" if speed_method == 'savgol' else ""))

    video_frames, _, _ = load_prediction_video(video_path)
    T, H, W = video_frames.shape

    print(f"  Shape: {T} frames × {H} × {W} pixels")

    transect_args = []
    for y_idx in range(H):
        timestack = extract_timestack(video_frames, y_idx)
        transect_args.append((y_idx, timestack, dt, dx, speed_method, savgol_window, savgol_polyorder, split_crossings))

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"  Processing {H} transects with {n_workers} workers...")

    # Parallelism note: we parallelize TRANSECTS within a video (this Pool), then do the
    # per-video post-processing (avi load, per-pixel aggregation, NetCDF, PNG) serially on
    # one core -> the other ~11 cores idle during that stretch, so average CPU load looks low.
    # A VIDEO-level Pool (36 independent videos run concurrently, transects serial within each)
    # would be ~2-3x faster overall: transect wall-time is ~unchanged (12-way video concurrency
    # ≈ 11-way transect concurrency, same total core-seconds), and it overlaps the otherwise-idle
    # serial post-processing across videos. Not done — the absolute runtime (~1 min/video) is fine.
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_transect, transect_args),
                           total=H, desc="  Transects",
                           disable=not sys.stdout.isatty()))   # no \r spam in log files

    if split_crossings:
        n_split = sum(r['split_stats'][0] for r in results)
        n_kept = sum(r['split_stats'][1] for r in results)
        pct = 100.0 * n_split / n_kept if n_kept else 0.0
        print(f"  split-crossings fired on {n_split}/{n_kept} components ({pct:.2f}%)")

    pixel_maps = build_per_pixel_speed_map(H, W, results)

    if save_videos:
        output_avi_path = output_dir / f"{video_path.stem}_speeds.avi"
        output_mp4_path = output_dir / f"{video_path.stem}_speeds.mp4"
        vmin, vmax = create_speed_videos(video_frames, pixel_maps,
                                        output_avi_path, output_mp4_path, fps_hz, dx)
    else:
        # Calculate vmin/vmax for the PNG map without creating videos
        speed_sum, _, counts, _, _ = pixel_maps
        with np.errstate(divide='ignore', invalid='ignore'):
            pixel_speeds = speed_sum / counts
        valid_speeds = pixel_speeds[counts > 0]
        vmin, vmax = (np.percentile(valid_speeds, 1), np.percentile(valid_speeds, 99)) if len(valid_speeds) > 0 else (0, 10)

    mean_speed_png = output_dir / f"{video_path.stem}_pixel_speed_map.png"
    create_mean_speed_map(H, W, pixel_maps, mean_speed_png, vmin, vmax)

    netcdf_path = output_dir / f"{video_path.stem}_speeds.nc"
    create_netcdf(H, W, pixel_maps, netcdf_path, video_path, video_frames, fps_hz)

    trajectory_csv = output_dir / f"{video_path.stem}_trajectories.csv"
    save_raw_trajectory_data(results, trajectory_csv, dt, dx)

def create_speed_videos(video_frames, pixel_maps, output_avi_path, output_mp4_path, fps, dx):
    """
    Create speed videos with individual wave speeds per pixel.
    Optimized version with pre-computed BGR and vectorized operations.

    AVI uses fixed scale of 0-10 m/s for consistent comparisons.
    MP4 uses percentile-based scale for better visualization.
    `pixel_maps` is the tuple returned by build_per_pixel_speed_map.
    """
    T, H, W = video_frames.shape

    # Per-pixel average speed map for coloring
    speed_sum, _, counts, _, _ = pixel_maps
    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_speeds = speed_sum / counts
        pixel_speeds[counts == 0] = np.nan

    # Fixed scale for AVI (0-10 m/s)
    avi_vmin, avi_vmax = 0, 10

    # Percentile-based scale for MP4 visualization
    valid_speeds = pixel_speeds[~np.isnan(pixel_speeds)]
    mp4_vmin, mp4_vmax = (np.percentile(valid_speeds, 1), np.percentile(valid_speeds, 99)) if len(valid_speeds) > 0 else (0, 10)

    # Pre-scale speeds to 0-255 for AVI (fixed 0-10 m/s scale)
    speeds_scaled = np.clip((pixel_speeds - avi_vmin) / (avi_vmax - avi_vmin + 1e-8) * 255, 0, 255)
    speeds_scaled = np.nan_to_num(speeds_scaled, nan=0).astype(np.uint8)
    
    # Pre-calculate BGR colors for the MP4 (convert once, not per frame!)
    cmap = plt.cm.jet
    speeds_norm = np.clip((pixel_speeds - mp4_vmin) / (mp4_vmax - mp4_vmin + 1e-8), 0, 1)
    speeds_norm = np.nan_to_num(speeds_norm, nan=0)
    pixel_colors_rgb = (cmap(speeds_norm)[:, :, :3] * 255).astype(np.uint8)
    # Pre-convert to BGR once
    pixel_colors_bgr = pixel_colors_rgb[:, :, ::-1]  # RGB to BGR
    
    fourcc_avi = cv2.VideoWriter_fourcc(*'HFYU')
    writer_avi = cv2.VideoWriter(str(output_avi_path), fourcc_avi, fps, (W, H), isColor=False)
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'avc1')
    writer_mp4 = cv2.VideoWriter(str(output_mp4_path), fourcc_mp4, fps, (W, H), isColor=True)
    
    # Pre-allocate frame buffers to avoid repeated allocation
    avi_frame = np.zeros((H, W), dtype=np.uint8)
    rgb_frame = np.zeros((H, W, 3), dtype=np.uint8)
    
    for t in tqdm(range(T), desc="  Writing videos", leave=False):
        mask = video_frames[t] > 0  # (H, W) boolean mask
        
        # --- AVI Frame (Grayscale Data) ---
        avi_frame.fill(0)  # Reset instead of recreating
        avi_frame[mask] = speeds_scaled[mask]
        writer_avi.write(avi_frame)
        
        # --- MP4 Frame (Visual BGR) ---
        rgb_frame.fill(0)  # Reset instead of recreating
        rgb_frame[mask] = pixel_colors_bgr[mask]  # Already in BGR!
        writer_mp4.write(rgb_frame)
    
    writer_avi.release()
    writer_mp4.release()
    print(f"  AVI scale: {avi_vmin}-{avi_vmax} m/s (fixed), MP4 scale: {mp4_vmin:.2f}-{mp4_vmax:.2f} m/s (adaptive)")
    return mp4_vmin, mp4_vmax  # Return MP4 scale for PNG map visualization

def create_colorbar_image(vmin, vmax, output_path):
    """Create a colorbar reference image."""
    fig, ax = plt.subplots(figsize=(8, 2))

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='jet', extent=[vmin, vmax, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel('Wave Speed (m/s)', fontsize=12)
    ax.set_title('Speed Colormap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Calculate wave speeds from prediction videos.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python calculate_wave_speeds.py video.avi output/
  python calculate_wave_speeds.py videos/ output/ --speed-method finite_diff
  python calculate_wave_speeds.py videos/ output/ --savgol-window 7 --save-videos
        '''
    )
    parser.add_argument('input', help='Input video file or folder containing *_pred.avi files')
    parser.add_argument('output', help='Output folder for results')
    parser.add_argument('--speed-method', choices=['savgol', 'finite_diff'], default='savgol',
                        help='Speed calculation method (default: savgol)')
    parser.add_argument('--savgol-window', type=int, default=5,
                        help='Savitzky-Golay window size, must be odd (default: 5)')
    parser.add_argument('--savgol-order', type=int, default=2,
                        help='Savitzky-Golay polynomial order (default: 2)')
    parser.add_argument('--save-videos', action='store_true',
                        help='Save AVI/MP4 speed visualization videos (slow)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Transect worker processes per video (default: CPU count minus one)')
    parser.add_argument('--completed-only', action='store_true',
                        help='For directory input, require the matching prediction density marker')
    parser.add_argument('--no-split-crossings', action='store_false', dest='split_crossings',
                        help='Disable crossing/parallel-crest separation and fall back to the '
                             'row-mean centroid. Splitting (per-row blobs + nearest-neighbour '
                             'association) is ON by default.')
    parser.set_defaults(split_crossings=True)
    args = parser.parse_args()
    if args.workers is not None and args.workers <= 0:
        parser.error('--workers must be positive')

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    space_res = 1.0  # meters per pixel

    print("="*80)
    print("WAVE SPEED CALCULATION PIPELINE")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Speed method: {args.speed_method}" +
          (f" (window={args.savgol_window}, order={args.savgol_order})" if args.speed_method == 'savgol' else ""))
    print("Time resolution: Date-dependent (2Hz default, 1Hz for 2021-09-19)")
    print(f"Space resolution: {space_res} m/pixel")
    print(f"Save videos: {args.save_videos}")
    print(f"Available CPUs: {cpu_count()}")
    print()

    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        # Skip exFAT/SMB AppleDouble sidecars (._*) that match the glob on SSDs.
        video_files = sorted(p for p in input_path.glob("*_pred.avi")
                             if not p.name.startswith("._"))
        if args.completed_only:
            video_files = [
                path for path in video_files
                if path.with_name(f"{path.stem}_density.png").is_file()
            ]
        if not video_files:
            print(f"No *_pred.avi files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Input path does not exist: {input_path}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s) to process")
    print()

    # Process each video. Continue after an error so a rerun can resume completed files,
    # then fail the command so callers do not mistake a partial run for success.
    failures = []
    for video_path in video_files:
        expected_outputs = (
            output_dir / f"{video_path.stem}_pixel_speed_map.png",
            output_dir / f"{video_path.stem}_speeds.nc",
            output_dir / f"{video_path.stem}_trajectories.csv",
        )
        if all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs):
            print(f"Skipping {video_path.name} (speed products already complete)")
            continue
        try:
            process_video(
                video_path, output_dir, dx=space_res,
                n_workers=args.workers,
                speed_method=args.speed_method,
                savgol_window=args.savgol_window,
                savgol_polyorder=args.savgol_order,
                save_videos=args.save_videos,
                split_crossings=args.split_crossings,
            )
        except Exception as e:
            failures.append(video_path.name)
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if failures:
        raise SystemExit(f"Wave-speed calculation failed for {len(failures)} video(s): {', '.join(failures)}")

    print("Wave-speed calculation complete.")


if __name__ == "__main__":
    main()
