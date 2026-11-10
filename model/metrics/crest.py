"""Quality-weighted and binary event-count crest matching for complete images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize


METRIC_VERSION = "crest_cldice_v4"
MIN_SKELETON_PIXELS = 5
DEFAULT_TOLERANCE = 2
DEFAULT_DETECTION_COVERAGE_THRESHOLD = 0.5


def _component_subtracks(
    component: np.ndarray,
    rows: np.ndarray,
    gate_px: float,
    max_gap: int = 3,
) -> list[list[tuple[int, np.ndarray]]]:
    """Split one component using the downstream tracker's row association."""
    active: list[dict] = []
    finished: list[dict] = []
    for row in rows:
        row_labels, n_blobs = label(component[row, :])
        blobs = [
            np.flatnonzero(row_labels == blob_id)
            for blob_id in range(1, n_blobs + 1)
        ]
        centroids = [float(blob.mean()) for blob in blobs]
        candidates = []
        for active_index, track in enumerate(active):
            if row - track["row"] <= max_gap:
                for blob_index, centroid in enumerate(centroids):
                    distance = abs(centroid - track["x"])
                    if distance <= gate_px:
                        candidates.append((distance, active_index, blob_index))
        used_tracks: set[int] = set()
        used_blobs: set[int] = set()
        for _, active_index, blob_index in sorted(candidates):
            if active_index in used_tracks or blob_index in used_blobs:
                continue
            track = active[active_index]
            track["points"].append((int(row), blobs[blob_index]))
            track["row"] = int(row)
            track["x"] = centroids[blob_index]
            used_tracks.add(active_index)
            used_blobs.add(blob_index)
        for blob_index, centroid in enumerate(centroids):
            if blob_index not in used_blobs:
                active.append({
                    "points": [(int(row), blobs[blob_index])],
                    "row": int(row),
                    "x": centroid,
                })
        still_active = []
        for track in active:
            (still_active if row - track["row"] <= max_gap else finished).append(track)
        active = still_active
    return [track["points"] for track in finished + active]


def _split_crest_tracks(
    mask: np.ndarray, gate_px: float = 12.0
) -> tuple[tuple[int, int], list[list[tuple[int, np.ndarray]]]]:
    mask = np.asarray(mask, dtype=bool)
    component_labels, n_components = label(
        mask, structure=np.ones((3, 3), dtype=bool)
    )
    tracks: list[list[tuple[int, np.ndarray]]] = []
    for component_id in range(1, n_components + 1):
        component = component_labels == component_id
        rows = np.flatnonzero(component.any(axis=1))
        tracks.extend(_component_subtracks(component, rows, gate_px))
    return mask.shape, tracks


@dataclass(frozen=True)
class EventFeature:
    """Raw mask, skeleton, and tolerance band for one reportable crest."""

    coordinates: np.ndarray
    band: np.ndarray
    band_origin: tuple[int, int]
    bbox: tuple[int, int, int, int]
    length: int
    mask_coordinates: np.ndarray
    mask_bbox: tuple[int, int, int, int]
    mask_area: int
    image_shape: tuple[int, int]


def _feature_from_coordinates(
    coordinates: np.ndarray,
    mask_coordinates: np.ndarray,
    image_shape: tuple[int, int],
    tolerance: int,
) -> EventFeature | None:
    if not len(coordinates):
        return None
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    height, width = image_shape
    crop_y0 = max(0, int(y0) - tolerance)
    crop_x0 = max(0, int(x0) - tolerance)
    crop_y1 = min(height, int(y1) + tolerance + 1)
    crop_x1 = min(width, int(x1) + tolerance + 1)
    local_skeleton = np.zeros(
        (crop_y1 - crop_y0, crop_x1 - crop_x0), dtype=bool
    )
    local_coordinates = coordinates - np.array((crop_y0, crop_x0))
    local_skeleton[local_coordinates[:, 0], local_coordinates[:, 1]] = True
    band = (
        binary_dilation(local_skeleton, iterations=tolerance)
        if tolerance
        else local_skeleton.copy()
    )
    return EventFeature(
        coordinates=coordinates,
        band=band,
        band_origin=(crop_y0, crop_x0),
        bbox=(int(y0), int(y1), int(x0), int(x1)),
        length=int(len(coordinates)),
        mask_coordinates=mask_coordinates,
        mask_bbox=(
            int(mask_coordinates[:, 0].min()),
            int(mask_coordinates[:, 0].max()),
            int(mask_coordinates[:, 1].min()),
            int(mask_coordinates[:, 1].max()),
        ),
        mask_area=int(len(mask_coordinates)),
        image_shape=(height, width),
    )


def _feature_from_track(
    track: list[tuple[int, np.ndarray]],
    image_shape: tuple[int, int],
    tolerance: int,
) -> EventFeature | None:
    rows = np.concatenate([
        np.full(len(columns), row, dtype=np.int64) for row, columns in track
    ])
    columns = np.concatenate([columns for _, columns in track]).astype(np.int64)
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(columns.min()), int(columns.max())
    # One zero-valued pixel of context reproduces full-image skeletonization
    # while making its cost proportional to the crest extent, not image area.
    crop_y0 = max(0, y0 - 1)
    crop_x0 = max(0, x0 - 1)
    crop_y1 = min(image_shape[0], y1 + 2)
    crop_x1 = min(image_shape[1], x1 + 2)
    local_event = np.zeros(
        (crop_y1 - crop_y0, crop_x1 - crop_x0), dtype=bool
    )
    local_event[rows - crop_y0, columns - crop_x0] = True
    local_coordinates = np.argwhere(skeletonize(local_event))
    local_coordinates += np.array((crop_y0, crop_x0))
    mask_coordinates = np.column_stack((rows, columns))
    return _feature_from_coordinates(
        local_coordinates, mask_coordinates, image_shape, tolerance
    )


def extract_event_features(
    mask: np.ndarray,
    tolerance: int = DEFAULT_TOLERANCE,
    min_skeleton_pixels: int = MIN_SKELETON_PIXELS,
) -> list[EventFeature]:
    """Split a mask and skeletonize each retained crest exactly once."""
    image_shape, tracks = _split_crest_tracks(mask)
    features = []
    for track in tracks:
        feature = _feature_from_track(track, image_shape, tolerance)
        if feature is not None and feature.length >= min_skeleton_pixels:
            features.append(feature)
    return features


def _bboxes_can_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    tolerance: int,
) -> bool:
    ay0, ay1, ax0, ax1 = first
    by0, by1, bx0, bx1 = second
    return not (
        ay1 + tolerance < by0
        or by1 + tolerance < ay0
        or ax1 + tolerance < bx0
        or bx1 + tolerance < ax0
    )


def _supported_count(coordinates: np.ndarray, feature: EventFeature) -> int:
    local = coordinates - np.asarray(feature.band_origin, dtype=coordinates.dtype)
    valid = (
        (local[:, 0] >= 0)
        & (local[:, 0] < feature.band.shape[0])
        & (local[:, 1] >= 0)
        & (local[:, 1] < feature.band.shape[1])
    )
    if not valid.any():
        return 0
    selected = local[valid]
    return int(feature.band[selected[:, 0], selected[:, 1]].sum())


def score_feature_matrix(
    predicted: list[EventFeature],
    labeled: list[EventFeature],
    tolerance: int = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return labeled-by-predicted pair clDice without full-image pair scans."""
    scores = np.zeros((len(labeled), len(predicted)), dtype=np.float32)
    for label_index, label_feature in enumerate(labeled):
        for pred_index, pred_feature in enumerate(predicted):
            if not _bboxes_can_overlap(
                label_feature.bbox, pred_feature.bbox, tolerance
            ):
                continue
            prediction_support = (
                _supported_count(pred_feature.coordinates, label_feature)
                / pred_feature.length
            )
            label_coverage = (
                _supported_count(label_feature.coordinates, pred_feature)
                / label_feature.length
            )
            denominator = prediction_support + label_coverage
            if denominator:
                scores[label_index, pred_index] = (
                    2.0 * prediction_support * label_coverage / denominator
                )
    return scores


def _raw_mask_band(
    feature: EventFeature,
    tolerance: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    y0, y1, x0, x1 = feature.mask_bbox
    height, width = feature.image_shape
    crop_y0 = max(0, y0 - tolerance)
    crop_x0 = max(0, x0 - tolerance)
    crop_y1 = min(height, y1 + tolerance + 1)
    crop_x1 = min(width, x1 + tolerance + 1)
    local_mask = np.zeros(
        (crop_y1 - crop_y0, crop_x1 - crop_x0), dtype=bool
    )
    local_coordinates = feature.mask_coordinates - np.array((crop_y0, crop_x0))
    local_mask[local_coordinates[:, 0], local_coordinates[:, 1]] = True
    band = (
        binary_dilation(local_mask, iterations=tolerance)
        if tolerance
        else local_mask
    )
    return band, (crop_y0, crop_x0)


def _coordinates_in_band(
    coordinates: np.ndarray,
    band: np.ndarray,
    origin: tuple[int, int],
) -> int:
    local = coordinates - np.asarray(origin, dtype=coordinates.dtype)
    valid = (
        (local[:, 0] >= 0)
        & (local[:, 0] < band.shape[0])
        & (local[:, 1] >= 0)
        & (local[:, 1] < band.shape[1])
    )
    if not valid.any():
        return 0
    selected = local[valid]
    return int(band[selected[:, 0], selected[:, 1]].sum())


def raw_mask_coverage_matrices(
    predicted: list[EventFeature],
    labeled: list[EventFeature],
    tolerance: int = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mutual directional coverage of raw masks within a tolerance."""
    shape = (len(labeled), len(predicted))
    prediction_support = np.zeros(shape, dtype=np.float32)
    target_coverage = np.zeros(shape, dtype=np.float32)
    predicted_bands = [_raw_mask_band(feature, tolerance) for feature in predicted]
    labeled_bands = [_raw_mask_band(feature, tolerance) for feature in labeled]
    for label_index, label_feature in enumerate(labeled):
        label_band, label_origin = labeled_bands[label_index]
        for pred_index, pred_feature in enumerate(predicted):
            if not _bboxes_can_overlap(
                label_feature.mask_bbox, pred_feature.mask_bbox, tolerance
            ):
                continue
            pred_band, pred_origin = predicted_bands[pred_index]
            prediction_support[label_index, pred_index] = (
                _coordinates_in_band(
                    pred_feature.mask_coordinates, label_band, label_origin
                )
                / pred_feature.mask_area
            )
            target_coverage[label_index, pred_index] = (
                _coordinates_in_band(
                    label_feature.mask_coordinates, pred_band, pred_origin
                )
                / label_feature.mask_area
            )
    return prediction_support, target_coverage


def maximum_quality_assignment(scores: np.ndarray) -> list[tuple[int, int]]:
    """Return the maximum-total-clDice one-to-one assignment."""
    scores = np.asarray(scores, dtype=np.float32)
    if not scores.size:
        return []
    label_indices, pred_indices = linear_sum_assignment(scores, maximize=True)
    return [
        (int(label_index), int(pred_index))
        for label_index, pred_index in zip(label_indices, pred_indices)
        if scores[label_index, pred_index] > 0.0
    ]


def maximum_mutual_coverage_assignment(
    prediction_support: np.ndarray,
    target_coverage: np.ndarray,
    min_coverage: float = 0.5,
) -> list[tuple[int, int]]:
    """Maximize binary matches satisfying both raw-mask coverage thresholds."""
    prediction_support = np.asarray(prediction_support, dtype=np.float32)
    target_coverage = np.asarray(target_coverage, dtype=np.float32)
    if not prediction_support.size:
        return []
    qualifies = (
        (prediction_support >= min_coverage)
        & (target_coverage >= min_coverage)
    )
    mean_coverage = (prediction_support + target_coverage) / 2.0
    cardinality_bonus = float(min(prediction_support.shape) + 1)
    utility = np.where(qualifies, cardinality_bonus + mean_coverage, 0.0)
    label_indices, pred_indices = linear_sum_assignment(utility, maximize=True)
    return [
        (int(label_index), int(pred_index))
        for label_index, pred_index in zip(label_indices, pred_indices)
        if qualifies[label_index, pred_index]
    ]


def soft_crest_counts(
    prediction: np.ndarray,
    target: np.ndarray | None = None,
    *,
    target_features: list[EventFeature] | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
    min_skeleton_pixels: int = MIN_SKELETON_PIXELS,
    detection_coverage_threshold: float = DEFAULT_DETECTION_COVERAGE_THRESHOLD,
) -> dict[str, int | float | str]:
    """Return binary crest counts and summed one-to-one pair quality.

    Supplying pre-extracted ``target_features`` avoids repeating target
    splitting, skeletonization, and dilation during operating-point sweeps.
    """
    predicted = extract_event_features(
        prediction, tolerance=tolerance, min_skeleton_pixels=min_skeleton_pixels
    )
    labeled = (
        target_features
        if target_features is not None
        else extract_event_features(
            target,
            tolerance=tolerance,
            min_skeleton_pixels=min_skeleton_pixels,
        )
    )
    scores = score_feature_matrix(predicted, labeled, tolerance=tolerance)
    prediction_support, target_coverage = raw_mask_coverage_matrices(
        predicted, labeled, tolerance=tolerance
    )
    quality_matches = maximum_quality_assignment(scores)
    detection_matches = maximum_mutual_coverage_assignment(
        prediction_support,
        target_coverage,
        detection_coverage_threshold,
    )
    quality = float(
        sum(float(scores[label_i, pred_i]) for label_i, pred_i in quality_matches)
    )
    matched = len(detection_matches)
    return {
        "crest_metric_version": METRIC_VERSION,
        "crest_predicted": len(predicted),
        "crest_labeled": len(labeled),
        "crest_quality_matched_pairs": len(quality_matches),
        "crest_quality_sum": quality,
        "crest_detection_tp": matched,
        "crest_unmatched_predicted": len(predicted) - matched,
        "crest_unmatched_labeled": len(labeled) - matched,
    }


def ratios(counts: dict[str, int | float | str]) -> dict[str, float]:
    """Convert additive counts into quality-weighted and event-count metrics."""
    quality = float(counts["crest_quality_sum"])
    predicted = int(counts["crest_predicted"])
    labeled = int(counts["crest_labeled"])
    matched = int(counts["crest_detection_tp"])
    unmatched_predicted = int(counts["crest_unmatched_predicted"])
    unmatched_labeled = int(counts["crest_unmatched_labeled"])
    return {
        "crest_precision": quality / predicted if predicted else float("nan"),
        "crest_recall": quality / labeled if labeled else float("nan"),
        "crest_f1": 2.0 * quality / (predicted + labeled)
        if predicted + labeled
        else float("nan"),
        "crest_detection_precision": matched / predicted
        if predicted
        else float("nan"),
        "crest_detection_recall": matched / labeled
        if labeled
        else float("nan"),
        "crest_detection_f1": 2.0 * matched / (predicted + labeled)
        if predicted + labeled
        else float("nan"),
        "crest_unmatched_prediction_rate": unmatched_predicted / predicted
        if predicted
        else float("nan"),
        "crest_missed_crest_rate": unmatched_labeled / labeled
        if labeled
        else float("nan"),
    }


def metadata() -> dict[str, int | float | str]:
    return {
        "crest_metric_version": METRIC_VERSION,
        "extraction": "directional_row_tracker_then_skeletonize",
        "min_skeleton_pixels": MIN_SKELETON_PIXELS,
        "spatial_tolerance_pixels": DEFAULT_TOLERANCE,
        "quality_assignment": "maximum_total_pair_cldice",
        "quality_aggregation": "quality_weighted",
        "detection_pair_score": "mutual_raw_mask_coverage_with_tolerance",
        "detection_coverage_threshold": DEFAULT_DETECTION_COVERAGE_THRESHOLD,
        "detection_spatial_tolerance_pixels": DEFAULT_TOLERANCE,
        "detection_criterion": "prediction_support_and_target_coverage_gte_threshold",
        "detection_assignment": "maximum_cardinality_then_maximum_mean_coverage",
        "detection_aggregation": "integer_tp_fp_fn",
    }
