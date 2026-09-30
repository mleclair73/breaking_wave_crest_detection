"""Binary segmentation metrics for crest masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, int]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = np.ones_like(target) if valid is None else np.asarray(valid, dtype=bool)
    prediction = prediction & valid
    target = target & valid
    return {
        "tp": int((prediction & target).sum()),
        "fp": int((prediction & ~target & valid).sum()),
        "fn": int((~prediction & target & valid).sum()),
        "tn": int((~prediction & ~target & valid).sum()),
    }


def binary_cldice(
    prediction: np.ndarray,
    target: np.ndarray,
    target_skeleton: np.ndarray | None = None,
) -> float:
    """Standard hard-mask clDice using exact morphological skeletons."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    pred_skeleton = skeletonize(prediction)
    target_skeleton = (
        skeletonize(target)
        if target_skeleton is None
        else np.asarray(target_skeleton, dtype=bool)
    )
    topology_precision = _ratio((pred_skeleton & target).sum(), pred_skeleton.sum())
    topology_recall = _ratio((target_skeleton & prediction).sum(), target_skeleton.sum())
    if not np.isfinite(topology_precision) or not np.isfinite(topology_recall):
        return 1.0 if not prediction.any() and not target.any() else 0.0
    return _ratio(
        2.0 * topology_precision * topology_recall,
        topology_precision + topology_recall,
    )


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    tolerance: int = 2,
    target_band: np.ndarray | None = None,
    target_skeleton: np.ndarray | None = None,
) -> dict[str, int | float]:
    """Compute table metrics and additive confusion counts for one image."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    valid = np.ones_like(target) if valid is None else np.asarray(valid, dtype=bool)
    prediction &= valid
    target &= valid
    counts = confusion_counts(prediction, target, valid)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    target_band = (
        binary_dilation(target, iterations=tolerance) if tolerance else target
    ) if target_band is None else np.asarray(target_band, dtype=bool)
    prediction_band = (
        binary_dilation(prediction, iterations=tolerance)
        if tolerance
        else prediction
    )
    boundary_precision = _ratio((prediction & target_band).sum(), prediction.sum())
    boundary_recall = _ratio((target & prediction_band).sum(), target.sum())
    if not np.isfinite(boundary_precision) or not np.isfinite(boundary_recall):
        boundary_f1 = 1.0 if not prediction.any() and not target.any() else 0.0
    else:
        boundary_f1 = _ratio(
            2.0 * boundary_precision * boundary_recall,
            boundary_precision + boundary_recall,
        )
    return {
        **counts,
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou_fg": _ratio(tp, tp + fp + fn),
        "pixel_precision": _ratio(tp, tp + fp),
        "pixel_recall": _ratio(tp, tp + fn),
        "boundary_f1": boundary_f1,
        "cldice": binary_cldice(prediction, target, target_skeleton),
    }
